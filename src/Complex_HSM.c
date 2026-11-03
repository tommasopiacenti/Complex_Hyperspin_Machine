//last update: 03/11/2026 — v2.9 (compute_H gating + trig-once restructure; per-region OpenMP retained)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <omp.h>
#include "Complex_HSM.h"

//Compile with: gcc -O3 -fopenmp -march=native src/Complex_HSM.c -o Complex_HSM.x -lgsl -lgslcblas -lm
//Run with: ./Complex_HSM.x <NS> <pth> <Complex_HSM.ini>

int main(int argc, char *argv[]) {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Input Read and Header
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //Int Variables Declaration
    int N,D,save_dyn,new_coupling,type_coup,initial_type,stop_at_E_target,stop_when_stable,steps_for_update,stable_window,save_steps,compute_eigenvalues,relative_H_change;
    int n_hramp;
    int lattice_d, lattice_NL, lattice_bc;
    int buffer_size, flush_interval, num_threads;
    int NS = atoi(argv[1]);

    //Double Variable Declaration
    double Amax,phim,dphi,g,beta,hm,dh,Dh,Dh2,deq,tmax,dt,t_coup,t_eq,t_pump,t_pump2,t_hramp,Dt_hramp,Dhramp,prob_coup,prob_pm_re,prob_pm_im,coup_phase,E_target,E_target_tol,allowed_H_change;
    double pth = atof(argv[2]);
    double t_final_value = 0;

    //Char Variable Declaration
    char output_dir[512] = "";
    char stop_reason[128] = "Reached tmax";

    //Check correct number of arguments when executing
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <NS> <pth> <Complex_HSM.ini>\n", argv[0]);
        return 1;
    }

    //Read parameters from ini file and put them in the corresponding variables
    if (parse_config(argv[3], &N, &D, &initial_type, &Amax, &phim, &dphi, &g, &beta, &hm, &dh, &Dh, &Dh2, &deq, &tmax, &dt, &t_coup, &t_eq, &t_pump, &t_pump2, &save_dyn, &new_coupling, &prob_coup, &prob_pm_re, &prob_pm_im, &coup_phase, &type_coup, &E_target, &E_target_tol, &stop_at_E_target, &stop_when_stable, &allowed_H_change, &steps_for_update, &stable_window, &relative_H_change, &buffer_size, &flush_interval, &num_threads, &save_steps, output_dir, sizeof(output_dir), &compute_eigenvalues, &lattice_d, &lattice_NL, &lattice_bc, &t_hramp, &n_hramp, &Dt_hramp, &Dhramp)) {
        return 1;
    }

    //Input Parameters Checks
    if (save_steps < 1) save_steps = 1;
    if (initial_type < 0 || initial_type > 2) {
        fprintf(stderr, "Error: initial_type must be 0 (real), 1 (pure imaginary), or 2 (general complex)\n");
        return 1;
    }

    if (n_hramp > 0 && Dt_hramp < dt) {
        fprintf(stderr, "Error: Dt_hramp (%.6f) must be >= dt (%.6f) when n_hramp > 0\n", Dt_hramp, dt);
        return 1;
    }

    // Map output_dir
    if (strcmp(output_dir, "1") == 0) {
        strcpy(output_dir, "results");
    } else if (strcmp(output_dir, "0") == 0) {
        output_dir[0] = '\0';
    }

    // Lattice mode: override N with NL^d
    if (new_coupling == 2) {
        int expected_N = 1;
        for (int i = 0; i < lattice_d; i++) expected_N *= lattice_NL;
        N = expected_N;
    }

    if (pth >= 0 && !compute_eigenvalues) {
        fprintf(stderr, "Error: pth >= 0 requires eigenvalue computation (compute_eigenvalues=1)\n");
        return 1;
    }

    print_header_and_ini_parameters(argv[3],N,D,initial_type,NS,Amax,phim,dphi,g,beta,hm,dh,Dh,Dh2,pth,deq,tmax,dt,t_coup,t_eq,t_pump,t_pump2,save_dyn,new_coupling,prob_coup,prob_pm_re,prob_pm_im,coup_phase,type_coup,E_target,E_target_tol,stop_at_E_target,stop_when_stable,allowed_H_change,steps_for_update,stable_window,relative_H_change,buffer_size,flush_interval,num_threads,save_steps,compute_eigenvalues,lattice_d,lattice_NL,lattice_bc,t_hramp,n_hramp,Dt_hramp,Dhramp);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Memory Allocation and Initialization
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

printf("> Initialization Started\n\n");

    //Ranom Generator Inizialization
    srand(NS);

    //System State Variable Declaration and Allocation
    system_state s;
    memset(&s, 0, sizeof(s));

    if (allocate_system(N, D, &s) != 0) {
        printf("\n\nMemory Allocation Failure\n\n");
        return 1;
    } else {
        printf("> Successful Memory Allocation\n\n");
    }

    //Observables Declaration
    double L_old = 0; //Dummy Variable for Derivative Computation
    observables obs = {0};

    initialize_fields(&s, Amax, initial_type);
    compute_hyperspin_norm(&s);

    if (new_coupling == 2) {
        // Lattice mode: allocate dense J only if compute_eigenvalues
        if (compute_eigenvalues) {
            s.J_re = calloc(N*N, sizeof(double));
            s.J_im = calloc(N*N, sizeof(double));
            if (!s.J_re || !s.J_im) { printf("\n\nDense J Memory Allocation Failure\n\n"); return 1; }
            s.J_allocated = 1;
        }
        initialize_J_Hermitian_Lattice(&s, lattice_d, lattice_NL, lattice_bc, prob_pm_re, prob_pm_im, prob_coup, type_coup);
    } else {
        // Always build sparse first (no pair_data, no dense J unless needed)
        if (new_coupling == 1) {
            initialize_J_Hermitian_Binary_Sparse(&s, prob_pm_re, prob_pm_im, prob_coup, type_coup, NS);
        } else {
            // Try sparse file first, fall back to dense for backward compatibility
            if (access("Data_Matrix_sparse.dat", F_OK) == 0)
                initialize_J_Sparse_From_Sparse_File(&s, "Data_Matrix_sparse.dat");
            else
                initialize_J_Sparse_From_File(&s, "Data_Matrix.dat");
        }
        // Optionally build dense J from sparse for eigenvalue computation
        if (compute_eigenvalues) {
            s.J_re = calloc(N*N, sizeof(double));
            s.J_im = calloc(N*N, sizeof(double));
            if (!s.J_re || !s.J_im) {
                printf("\n\nDense J Memory Allocation Failure\n\n");
                return 1;
            }
            s.J_allocated = 1;
            fill_dense_from_sparse(&s);
            // Write dense Data_Matrix.dat for reproducibility
            write_dense_matrix_file(&s, "Data_Matrix.dat");
        }
    }
    // Apply coupling rotation if nonzero (post-matrix-construction)
    apply_coupling_rotation(&s, coup_phase);

    // Re-write matrix files to persist rotated state (only when rotation was applied)
    if (coup_phase != 0.0) {
        write_sparse_matrix_file(&s, "Data_Matrix_sparse.dat");
        if (s.J_allocated) write_dense_matrix_file(&s, "Data_Matrix.dat");
    }

    // Compute hash and create output directory
    char hash_hex[17] = "";
    char result_dir[1024] = "";
    if (output_dir[0]) {
        mkdir(output_dir, 0755);
        // Hash from whatever matrix file exists (sparse preferred)
        const char *matrix_file = "Data_Matrix_sparse.dat";
        if (access(matrix_file, F_OK) != 0) matrix_file = "Data_Matrix.dat";
        unsigned long long h = compute_hash(argv[3], NS, matrix_file);
        for (const char *p = argv[2]; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
        format_hash_hex(h, hash_hex);
        snprintf(result_dir, sizeof(result_dir), "%s/run_%s", output_dir, hash_hex);
        mkdir(result_dir, 0755);
        copy_file_to_dir(argv[3], result_dir, "Complex_HSM.ini");
        // Copy whichever matrix files exist
        if (access("Data_Matrix_sparse.dat", F_OK) == 0)
            copy_file_to_dir("Data_Matrix_sparse.dat", result_dir, "Data_Matrix_sparse.dat");
        if (access("Data_Matrix.dat", F_OK) == 0)
            copy_file_to_dir("Data_Matrix.dat", result_dir, "Data_Matrix.dat");
        printf("> Output directory: %s\n\n", result_dir);
    }

    if (pth >= 0) {
        printf("> Computing the pump threshold 'hth'\n");
        double hth = h_threshold(N, g, s.J_re, s.J_im);
        hm = hth * pth;
        printf("> hm --> %0.4f\n\n", hm);
    }

    initialize_pump(&s, s.h, hm, dh);
    initialize_pump(&s, s.phi, phim, dphi);
    precompute_trig(&s);

    if (compute_eigenvalues) {
        //hmod is used at this point to just have a zero pump in the computation of the J max eigenvalue.
        initialize_pump(&s, s.hmod, 0, 0);

        //Max eigenvalue computation for coupling matrix J
        double EigenJ_max = Linearized_Max_Eigenvalue(N, s.J_re, s.J_im, s.hmod, 0);
        printf("> Jmax = %0.4f  (max coupling matrix eigenvalue)\n\n", EigenJ_max);

        //Full linearized at the origin problem max eigenvalue computation
        double EigenL_max = Linearized_Max_Eigenvalue(N, s.J_re, s.J_im, s.h, g);
        printf("> Lmax = %0.4f  (max Linearized matrix eigenvalue with 'dh')\n\n", EigenL_max);

    } else {
        printf("> Jmax = N/A  (eigenvalue computation disabled)\n\n");
        printf("> Lmax = N/A  (eigenvalue computation disabled)\n\n");
    }

    //prepare pump modification array
    initialize_pump(&s, s.hmod, Dh, 0);

    printf("> Initializing output files (buffer_size=%d, flush_interval=%d)\n", buffer_size, flush_interval);
    io_context ctx = init_output_files(N, D, save_dyn, buffer_size, result_dir);
    if (ctx.f_final == NULL && save_dyn == 0) {
        printf("\n\nOutput File Initialization Failure\n\n");
        return 1;
    }

    printf("> Initialization Completed\n\n");
    print_memory_peak(&s);

    printf("------------------------------------------------------------------------\n");
    printf("------------------------------------------------------------------------\n\n");

// Set OpenMP thread count at runtime
    if (num_threads > 1) {
        omp_set_num_threads(num_threads);
        use_parallel = 1;
    }

// Runge Kutta
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double t = 0.0;
int steps = 0;
int coup_control = 0;
int eq_control = 0;
int pump_control = 0;
int pump2_control = 0;
int ramp_fired = 0;
int stable_count = 0;
int first_stable_check = 1;
double H_check_old = 0.0;
int t_final_printed = 0;

double t_ramp_end = (n_hramp > 0) ? t_hramp + (n_hramp - 1) * Dt_hramp : -1.0;
double t_last_change = t_coup;
if (t_eq > t_last_change)   t_last_change = t_eq;
if (t_pump > t_last_change) t_last_change = t_pump;
if (t_pump2 > t_last_change) t_last_change = t_pump2;
if (t_ramp_end > t_last_change) t_last_change = t_ramp_end;

printf("> Runge-Kutta Integration Started\n\n");

while (t < tmax) {

    printf("\r> Time Progress: %d %%\r",(int)(t*100/tmax+1));

    if (coup_control == 0 && t >= t_coup && t_coup >= 0) { coup_control = 1; }
    if (eq_control == 0 && t >= t_eq && t_eq >= 0) { eq_control = 1; }

    if (pump_control == 0 && t >= t_pump && t_pump >= 0) { modify_pump(&s); pump_control = 1; }
    if (pump2_control == 0 && t >= t_pump2 && t_pump2 >= 0) { for (int i = 0; i < N; i++) s.h[i] += Dh2; pump2_control = 1; }
    if (ramp_fired < n_hramp && t_hramp >= 0) {
        double next_fire = t_hramp + ramp_fired * Dt_hramp;
        if (t >= next_fire) { ramp_pump(&s, Dhramp); ramp_fired++; }
    }

    // Compute H only on save, E-target, stability-check, or final steps; full observables on save steps
    int save_step = (save_dyn > 0 && steps % save_steps == 0) ? 1 : 0;
    int compute_H = (save_step == 1 || stop_at_E_target == 1 ||
                     (stop_when_stable == 1 && (steps + 1) % steps_for_update == 0) ||
                     (t + dt >= tmax)) ? 1 : 0;
    if (compute_H || save_step)
        obs = compute_observables(&s, g, beta, dt, &L_old, compute_H, save_step);
    else
        obs = (observables){0};

    if (save_step) {
        save_output_files(&ctx, save_dyn, &s, t, obs.L, obs.L0, obs.Lc, obs.dL, obs.H, obs.m, obs.DS2);
    }

    if (steps % flush_interval == 0 && save_dyn > 0) {
        flush_output_files(&ctx, save_dyn);
    }

    RK2_step(&s, dt, g, beta, deq, coup_control, eq_control);

    t += dt;
    steps += 1;

    if (stop_at_E_target == 1 && obs.H <= E_target * (1 - E_target_tol)) {
        printf("\n\n> Stopped (Target Energy Reached)\n");
        printf("> t_final = %0.4f\n", t);
        t_final_printed = 1;
        t_final_value = t;
        strcpy(stop_reason, "Target energy reached");
        t = tmax;
    }

    // Check stability condition
    if (stop_when_stable == 1 && steps % steps_for_update == 0 && t > t_last_change) {
        if (first_stable_check) {
            H_check_old = obs.H;
            first_stable_check = 0;
        } else {
            double H_stop_tol = allowed_H_change;
            if (relative_H_change) H_stop_tol *= fabs(obs.H);
            if (fabs(obs.H - H_check_old) <= H_stop_tol) {
                stable_count++;
            } else {
                stable_count = 0;
            }

            if (stable_count >= stable_window) {
                printf("\n\n> Stopped (Energy stable for %d consecutive checks)\n", stable_window);
                printf("> t_final = %0.4f\n", t);
                t_final_printed = 1;
                t_final_value = t;
                strcpy(stop_reason, "Energy stable");
                t = tmax;
            }

            H_check_old = obs.H;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Save Final Configuration, Free Memory and Close, Print Final Energy
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if (!t_final_printed) {
    printf("\n\n> Stopped (Reached tmax)\n");
    printf("> t_final = %0.4f\n", t);
    t_final_value = t;
}

save_final_config(&ctx, &s, t, NS);

if (result_dir[0]) {
    write_metadata(result_dir, NS, N, D, initial_type, pth, hm, obs.H, t_final_value, stop_reason);
    printf("> Results saved in: %s\n\n", result_dir);
}

if (save_dyn > 0) {
    close_output_files(&ctx, save_dyn);
} else {
    if (ctx.f_final) fclose(ctx.f_final);
}

printf("\n\n> H_final = %0.15f \n\n", obs.H);

if(system_is_real(&s)){printf("> The system's final state is Complex\n\n");}
else{printf("> The system's final state is Real\n\n");}

printf("-----------------------------------------------------------------------\n");
printf("-----------------------------------------------------------------------\n");

free_system(&s);
}
