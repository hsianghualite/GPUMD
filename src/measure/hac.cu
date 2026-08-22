/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Calculate the heat current autocorrelation (HAC) function.
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "hac.cuh"
#include "integrate/ensemble_qct.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200
#define DIM 3

// Allocate memory for recording heat current data
void HAC::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute) {
    const auto* qct = dynamic_cast<const Ensemble_QCT*>(integrate.ensemble.get());
    if (qct != nullptr && qct->is_batch()) {
      batch_mode_ = true;
      number_of_replicas_ = qct->number_of_replicas();
      atoms_per_replica_ = qct->atoms_per_replica();
    } else {
      batch_mode_ = false;
      number_of_replicas_ = 1;
      atoms_per_replica_ = atom.number_of_atoms;
    }
    if (number_of_replicas_ <= 0 || atoms_per_replica_ <= 0) {
      PRINT_INPUT_ERROR("HAC received an invalid QCT replica layout.");
    }

    normalized_weights_.assign(number_of_replicas_, 1.0 / number_of_replicas_);
    if (qct != nullptr) {
      const std::vector<double> log_weights = qct->replica_log_wigner_weights();
      if (static_cast<int>(log_weights.size()) != number_of_replicas_) {
        PRINT_INPUT_ERROR("QCT Wigner weight count does not match HAC replicas.");
      }
      double maximum = -std::numeric_limits<double>::infinity();
      for (const double log_weight : log_weights) {
        if (std::isnan(log_weight) || log_weight == std::numeric_limits<double>::infinity()) {
          PRINT_INPUT_ERROR("QCT Wigner weights contain NaN or positive infinity.");
        }
        if (std::isfinite(log_weight)) {
          maximum = std::max(maximum, log_weight);
        }
      }
      if (!std::isfinite(maximum)) {
        PRINT_INPUT_ERROR("All QCT Wigner weights are zero.");
      }
      double total = 0.0;
      for (int replica = 0; replica < number_of_replicas_; ++replica) {
        normalized_weights_[replica] =
          std::isfinite(log_weights[replica]) ? std::exp(log_weights[replica] - maximum) : 0.0;
        total += normalized_weights_[replica];
      }
      if (!std::isfinite(total) || total <= 0.0) {
        PRINT_INPUT_ERROR("QCT Wigner weights have an invalid normalized total.");
      }
      for (double& weight : normalized_weights_) {
        weight /= total;
      }
      if (batch_mode_) {
        std::ofstream weights_output("hac_reweighting.csv");
        if (!weights_output.is_open()) {
          PRINT_INPUT_ERROR("Cannot open hac_reweighting.csv for writing.");
        }
        weights_output << "replica,seed,log_wigner_weight,normalized_weight\n";
        const auto& seeds = qct->replica_seeds();
        for (int replica = 0; replica < number_of_replicas_; ++replica) {
          const double log_weight = log_weights[replica];
          weights_output << replica << ','
                         << (replica < static_cast<int>(seeds.size()) ? seeds[replica] : 0ULL)
                         << ',' << log_weight << ',' << normalized_weights_[replica] << '\n';
        }
      }
    }

    const size_t number_of_frames = static_cast<size_t>(number_of_steps / sample_interval);
    const size_t number_of_values =
      static_cast<size_t>(NUM_OF_HEAT_COMPONENTS) * number_of_replicas_ * number_of_frames;
    if (number_of_frames == 0 ||
        number_of_values > std::numeric_limits<size_t>::max() / sizeof(double)) {
      PRINT_INPUT_ERROR("HAC data size is invalid or exceeds addressable memory.");
    }
    heat_all.resize(number_of_values);
    atom.heat_per_atom.resize(atom.number_of_atoms * 5);
  }
}

// sum up the per-atom heat current to get the total heat current
static __global__ void gpu_sum_heat(
  const int atoms_per_replica,
  const int total_atoms,
  const int Nd,
  const int nd,
  const int number_of_replicas,
  const double* g_heat,
  double* g_heat_all)
{
  // <<<NUM_OF_HEAT_COMPONENTS * number_of_replicas, 1024>>>
  const int tid = threadIdx.x;
  const int block = blockIdx.x;
  const int replica = block / NUM_OF_HEAT_COMPONENTS;
  const int component = block % NUM_OF_HEAT_COMPONENTS;
  const int number_of_patches = (atoms_per_replica - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < atoms_per_replica && replica < number_of_replicas) {
      const int atom_index = replica * atoms_per_replica + n;
      s_data[tid] += g_heat[atom_index + total_atoms * component];
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all[nd + Nd * block] = s_data[0];
  }
}

// sample heat current data for HAC calculations.
void HAC::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (!compute)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  compute_heat(atom.virial_per_atom, atom.velocity_per_atom, atom.heat_per_atom);

  int nd = (step + 1) / sample_interval - 1;
  int Nd = number_of_steps / sample_interval;
  gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS * number_of_replicas_, 1024>>>(
    atoms_per_replica_,
    atom.number_of_atoms,
    Nd,
    nd,
    number_of_replicas_,
    atom.heat_per_atom.data(),
    heat_all.data());
  GPU_CHECK_KERNEL
}

// Calculate the Heat current Auto-Correlation function (HAC)
static __global__ void gpu_find_hac(
  const int Nc,
  const int Nd,
  const int number_of_replicas,
  const double* g_heat,
  double* g_hac)
{
  //<<<dim3(Nc, number_of_replicas), 128>>>

  __shared__ double s_hac_xi[128];
  __shared__ double s_hac_xo[128];
  __shared__ double s_hac_yi[128];
  __shared__ double s_hac_yo[128];
  __shared__ double s_hac_z[128];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int replica = blockIdx.y;
  int number_of_patches = (Nd - 1) / 128 + 1;
  int number_of_data = Nd - bid;
  const int heat_offset = replica * NUM_OF_HEAT_COMPONENTS * Nd;
  const int hac_offset = replica * NUM_OF_HEAT_COMPONENTS * Nc;

  s_hac_xi[tid] = 0.0;
  s_hac_xo[tid] = 0.0;
  s_hac_yi[tid] = 0.0;
  s_hac_yo[tid] = 0.0;
  s_hac_z[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int index = tid + patch * 128;
    if (index + bid < Nd) {
      s_hac_xi[tid] += g_heat[heat_offset + index + Nd * 0] * g_heat[heat_offset + index + bid + Nd * 0] +
                       g_heat[heat_offset + index + Nd * 0] * g_heat[heat_offset + index + bid + Nd * 1];
      s_hac_xo[tid] += g_heat[heat_offset + index + Nd * 1] * g_heat[heat_offset + index + bid + Nd * 1] +
                       g_heat[heat_offset + index + Nd * 1] * g_heat[heat_offset + index + bid + Nd * 0];
      s_hac_yi[tid] += g_heat[heat_offset + index + Nd * 2] * g_heat[heat_offset + index + bid + Nd * 2] +
                       g_heat[heat_offset + index + Nd * 2] * g_heat[heat_offset + index + bid + Nd * 3];
      s_hac_yo[tid] += g_heat[heat_offset + index + Nd * 3] * g_heat[heat_offset + index + bid + Nd * 3] +
                       g_heat[heat_offset + index + Nd * 3] * g_heat[heat_offset + index + bid + Nd * 2];
      s_hac_z[tid] += g_heat[heat_offset + index + Nd * 4] * g_heat[heat_offset + index + bid + Nd * 4];
    }
  }
  __syncthreads();


  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_hac_xi[tid] += s_hac_xi[tid + offset];
      s_hac_xo[tid] += s_hac_xo[tid + offset];
      s_hac_yi[tid] += s_hac_yi[tid + offset];
      s_hac_yo[tid] += s_hac_yo[tid + offset];
      s_hac_z[tid] += s_hac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_hac[hac_offset + bid + Nc * 0] = s_hac_xi[0] / number_of_data;
    g_hac[hac_offset + bid + Nc * 1] = s_hac_xo[0] / number_of_data;
    g_hac[hac_offset + bid + Nc * 2] = s_hac_yi[0] / number_of_data;
    g_hac[hac_offset + bid + Nc * 3] = s_hac_yo[0] / number_of_data;
    g_hac[hac_offset + bid + Nc * 4] = s_hac_z[0] / number_of_data;
  }
}

// Calculate the Running Thermal Conductivity (RTC) from the HAC
static void find_rtc(const int Nc, const double factor, const double* hac, double* rtc)
{
  for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
    for (int nc = 1; nc < Nc; nc++) {
      const int index = Nc * k + nc;
      rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index]) * factor;
    }
  }
}

// Calculate HAC (heat currant auto-correlation function)
// and RTC (running thermal conductivity)
void HAC::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute)
    return;
  if (temperature <= 0.0) {
    PRINT_INPUT_ERROR(
      "compute_hac requires a positive temperature; QCT/LSC-IVR T=0 is "
      "incompatible with Green-Kubo thermal conductivity.");
  }
  print_line_1();
  printf("Start to calculate HAC and related quantities.\n");

  const int Nd = number_of_steps / sample_interval;
  const double dt = time_step * sample_interval;
  const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

  // Keep HAC and RTC per replica until the weighted reduction. This prevents
  // cross-replica heat-current terms from entering a native QCT batch.
  const size_t per_replica_size = static_cast<size_t>(Nc) * NUM_OF_HEAT_COMPONENTS;
  const size_t total_size = per_replica_size * number_of_replicas_;
  std::vector<double> rtc(total_size, 0.0);
  GPU_Vector<double> hac_gpu(total_size);
  std::vector<double> hac_cpu(total_size);
  std::vector<double> hac_merged(per_replica_size, 0.0);
  std::vector<double> rtc_merged(per_replica_size, 0.0);

  // Here, the block size is fixed to 128, which is a good choice
  dim3 hac_grid(Nc, number_of_replicas_);
  gpu_find_hac<<<hac_grid, 128>>>(Nc, Nd, number_of_replicas_, heat_all.data(), hac_gpu.data());
  GPU_CHECK_KERNEL

  hac_gpu.copy_to_host(hac_cpu.data());

  double factor = dt * 0.5 / (K_B * temperature * temperature * box.get_volume());
  factor *= KAPPA_UNIT_CONVERSION;

  for (int replica = 0; replica < number_of_replicas_; ++replica) {
    const size_t offset = static_cast<size_t>(replica) * per_replica_size;
    find_rtc(Nc, factor, hac_cpu.data() + offset, rtc.data() + offset);
    for (size_t index = 0; index < per_replica_size; ++index) {
      hac_merged[index] += normalized_weights_[replica] * hac_cpu[offset + index];
      rtc_merged[index] += normalized_weights_[replica] * rtc[offset + index];
    }
  }

  double weight_square_sum = 0.0;
  double max_weight = 0.0;
  for (const double weight : normalized_weights_) {
    weight_square_sum += weight * weight;
    max_weight = std::max(max_weight, weight);
  }
  const double effective_replicas = weight_square_sum > 0.0 ? 1.0 / weight_square_sum : 0.0;
  printf(
    "HAC Wigner reduction: replicas=%d N_eff=%.6g max_normalized_weight=%.6g\n",
    number_of_replicas_,
    effective_replicas,
    max_weight);

  FILE* fid = fopen("hac.out", "a");
  if (fid == nullptr) {
    PRINT_INPUT_ERROR("Cannot open hac.out for writing.");
  }
  FILE* replica_fid = nullptr;
  if (batch_mode_) {
    replica_fid = fopen("hac_replica.out", "a");
    if (replica_fid == nullptr) {
      fclose(fid);
      PRINT_INPUT_ERROR("Cannot open hac_replica.out for writing.");
    }
  }
  const int number_of_output_data = Nc / output_interval;
  for (int nd = 0; nd < number_of_output_data; nd++) {
    const int nc = nd * output_interval;
    double hac_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    double rtc_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
      for (int m = 0; m < output_interval; m++) {
        const int count = Nc * k + nc + m;
        hac_ave[k] += hac_merged[count];
        rtc_ave[k] += rtc_merged[count];
      }
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      hac_ave[m] /= output_interval;
      rtc_ave[m] /= output_interval;
    }
    if (replica_fid != nullptr) {
      for (int replica = 0; replica < number_of_replicas_; ++replica) {
        const size_t offset = static_cast<size_t>(replica) * per_replica_size;
        fprintf(replica_fid, "%d %25.15e", replica, (nc + output_interval * 0.5) * dt_in_ps);
        for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; ++k) {
          double value = 0.0;
          for (int m = 0; m < output_interval; ++m) {
            value += hac_cpu[offset + Nc * k + nc + m];
          }
          fprintf(replica_fid, "%25.15e", value / output_interval);
        }
        for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; ++k) {
          double value = 0.0;
          for (int m = 0; m < output_interval; ++m) {
            value += rtc[offset + Nc * k + nc + m];
          }
          fprintf(replica_fid, "%25.15e", value / output_interval);
        }
        fprintf(replica_fid, "\n");
      }
    }
    fprintf(fid, "%25.15e", (nc + output_interval * 0.5) * dt_in_ps);
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", hac_ave[m]);
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", rtc_ave[m]);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);
  if (replica_fid != nullptr) {
    fflush(replica_fid);
    fclose(replica_fid);
  }

  printf("HAC and related quantities are calculated.\n");
  print_line_2();

  compute = 0;
}

void HAC::parse(const char** param, int num_param)
{
  compute = 1;

  printf("Compute HAC.\n");

  if (num_param != 4) {
    PRINT_INPUT_ERROR("compute_hac should have 3 parameters.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for HAC should be an integer number.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for HAC should be an integer number.\n");
  }
  printf("    Nc is %d\n", Nc);

  if (!is_valid_int(param[3], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HAC should be an integer number.\n");
  }
  printf("    output_interval is %d\n", output_interval);
}

HAC::HAC(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "compute_hac";
}
