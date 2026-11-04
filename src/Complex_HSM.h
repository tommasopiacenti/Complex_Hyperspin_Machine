//last update: 03/11/2026 - v2.9 (compute_H gating + trig-once restructure; per-region OpenMP retained)

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/resource.h>
#include <malloc.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STRUCT TYPES
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//I/O context - holds all open file handles for buffered output
typedef struct {
    FILE *f_fields;
    FILE *f_equalizers;
    FILE *f_hamilton;
    FILE *f_lyapunov;
    FILE *f_h_ly_m_ds2;
    FILE *f_final;
} io_context;

//System state - flat contiguous arrays instead of pointer chains
//All arrays are 1D, indexed via macros defined below
typedef struct {
    int N;            // Number of spins
    int D;            // Number of components per spin
    double *S_re;     // Hyperspin real parts, size N*D
    double *S_im;     // Hyperspin imaginary parts, size N*D
    double *S2;       // Squared norms per spin, size N
    double *DS_re;    // Midpoint hyperspin real parts, size N*D
    double *DS_im;    // Midpoint hyperspin imaginary parts, size N*D
    double *DS2;      // Midpoint squared norms per spin, size N
    double *Y_re;     // Equalizer real parts, size N*2
    double *Y_im;     // Equalizer imaginary parts, size N*2
    double *DY_re;    // Midpoint equalizer real parts, size N*2
    double *DY_im;    // Midpoint equalizer imaginary parts, size N*2
    double *J_re;     // Coupling matrix real parts, size N*N
    double *J_im;     // Coupling matrix imaginary parts, size N*N
    double *h;        // Pump values, size N
    double *hmod;     // First-stage pump modification, size N

    double *phi;      // Phase values, size N
    double *JS_re;    // J*S coupling product real parts, size N*D
    double *JS_im;    // J*S coupling product imaginary parts, size N*D
    // Quick-win scratch arrays (allocated once, reused every timestep)
    double *cos_phi;  // Precomputed cos(phi[a]), size N
    double *sin_phi;  // Precomputed sin(phi[a]), size N
    double *inv_sqrtS2; // Reusable scratch buffer, size N
    // Sparse coupling structures (built once, used in hot loops)
    int    *csr_row_ptr;
    int    *csr_col_idx;
    double *csr_Jr;
    double *csr_Ji;
    int     num_edges;
    int    *edge_a;
    int    *edge_b;
    double *edge_Jr;
    double *edge_Ji;
    int J_allocated;   // 1 if dense J_re/J_im were allocated (for eigenvalue computation)
    unsigned long startup_heap_B;
    unsigned long steady_heap_B;
} system_state;

// Global flag: set to 0 when num_threads <= 1 to skip all OpenMP overhead
int use_parallel = 0;

// Observable bundle - single function for all diagnostics
typedef struct {
    double H;           // always valid
    double L, L0, Lc;   // full set valid only when compute_full=1
    double dL;
    double m, DS2;
} observables;

// Memory usage logging helper (negligible overhead, call once at peak)
static inline void print_memory_peak(system_state *s) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    struct mallinfo2 mi = mallinfo2();
    s->steady_heap_B = (unsigned long)(mi.uordblks + mi.hblkhd);
    printf("> [MEM] startup: heap=%lu bytes, steady: heap=%lu bytes, RSS=%ld KB\n",
           s->startup_heap_B, s->steady_heap_B, ru.ru_maxrss);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//INDEX MACROS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Hyperspin component access: field (a, j) where a=spin, j=component
#define S_RE(s, a, j)  (s)->S_re[(a)*(s)->D + (j)]
#define S_IM(s, a, j)  (s)->S_im[(a)*(s)->D + (j)]
#define S2(s, a)       (s)->S2[a]
#define DS_RE(s, a, j) (s)->DS_re[(a)*(s)->D + (j)]
#define DS_IM(s, a, j) (s)->DS_im[(a)*(s)->D + (j)]
#define DS2(s, a)      (s)->DS2[a]

// Equalizer access: field (a, k) where k=0,1
#define Y_RE(s, a, k)  (s)->Y_re[(a)*2 + (k)]
#define Y_IM(s, a, k)  (s)->Y_im[(a)*2 + (k)]
#define DY_RE(s, a, k) (s)->DY_re[(a)*2 + (k)]
#define DY_IM(s, a, k) (s)->DY_im[(a)*2 + (k)]

// Coupling matrix access
#define J_RE(s, a, b)  (s)->J_re[(a)*(s)->N + (b)]
#define J_IM(s, a, b)  (s)->J_im[(a)*(s)->N + (b)]

// Precomputed coupling product
#define JS_RE(s, a, j) (s)->JS_re[(a)*(s)->D + (j)]
#define JS_IM(s, a, j) (s)->JS_im[(a)*(s)->D + (j)]

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//FUNCTIONS DECLARATIONS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Header & Input Feedback
void print_header_and_ini_parameters(const char *filename, int N, int D, int initial_type, int NS, double Amax, double phim, double dphi, double g, double beta, double hm, double dh, double Dh, double Dh2, double pth, double deq, double tmax, double dt, double t_coup, double t_eq, double t_pump, double t_pump2, int save_dyn, int new_coupling, double prob_coup, double prob_pm_re, double prob_pm_im, double coup_phase, int type_coup, double E_target, double E_target_tol, int stop_at_E_target, int stop_when_stable, double allowed_H_change, int steps_for_update, int stable_window, int relative_H_change, int buffer_size, int flush_interval, int num_threads, int save_steps, int compute_eigenvalues, int lattice_d, int lattice_NL, int lattice_bc, double t_hramp, int n_hramp, double Dt_hramp, double Dhramp);

//Input
int parse_config(const char *filename, int *N, int *D, int *initial_type, double *Amax, double *phim, double *dphi, double *g, double *beta, double *hm, double *dh, double *Dh, double *Dh2, double *deq, double *tmax, double *dt, double *t_coup, double *t_eq, double *t_pump, double *t_pump2, int *save_dyn, int *new_coupling, double *prob_coup, double *prob_pm_re, double *prob_pm_im, double *coup_phase, int *type_coup, double *E_target, double *E_target_tol, int *stop_at_E_target, int *stop_when_stable, double *allowed_H_change, int *steps_for_update, int *stable_window, int *relative_H_change, int *buffer_size, int *flush_interval, int *num_threads, int *save_steps, char *output_dir, size_t output_dir_sz, int *compute_eigenvalues, int *lattice_d, int *lattice_NL, int *lattice_bc, double *t_hramp, int *n_hramp, double *Dt_hramp, double *Dhramp);

//System state management
int allocate_system(int N, int D, system_state *s);
void free_system(system_state *s);

//Initialization
void initialize_fields(system_state *s, double Amax, int initial_type);
void initialize_J_Hermitian_Binary_Random(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup);
void initialize_J_Read_From_File(system_state *s, const char *filename);
void initialize_pump(system_state *s, double *target, double hm, double dh);
void modify_pump(system_state *s);
void ramp_pump(system_state *s, double Dhramp);

void precompute_trig(system_state *s);
void build_sparse_coupling(system_state *s);
void initialize_J_Hermitian_Binary_Sparse(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup, int NS);
void initialize_J_Sparse_From_File(system_state *s, const char *filename);
void initialize_J_Sparse_From_Sparse_File(system_state *s, const char *filename);
void initialize_J_Hermitian_Lattice(system_state *s, int lattice_d, int lattice_NL, int lattice_bc, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup);
void write_sparse_matrix_file(system_state *s, const char *filename);
void write_dense_matrix_file(system_state *s, const char *filename);
void fill_dense_from_sparse(system_state *s);
void apply_coupling_rotation(system_state *s, double coup_phase);

//Update functions
void compute_hyperspin_norm(system_state *s);
void compute_coupling_product_flat(const system_state *s, const double *src_re, const double *src_im, double *out_re, double *out_im);
void RK2_step(system_state *s, double dt, double g, double beta, double deq, int coup_control, int eq_control);

//Physical functions
double S2_of_spin(system_state *s, int a);
double Y2eq_of_spin(system_state *s, int a);
double max_eigenvalue_symm(double *data, int n);
double Linearized_Max_Eigenvalue(int N, double *J_re, double *J_im, double *h, double g);
double h_threshold(int N, double g, double *J_re, double *J_im);
observables compute_observables(system_state *s, double g, double beta, double dt, double *L_old, int compute_H, int compute_full);
int system_is_real(system_state *s);

//Output
io_context init_output_files(int N, int D, int save_dyn, int buffer_size, const char *base_dir);
void save_output_files(io_context *ctx, int save_dyn, system_state *s, double t, double L, double L0, double Lc, double dL, double H, double m, double DS2);
void flush_output_files(io_context *ctx, int save_dyn);
void close_output_files(io_context *ctx, int save_dyn);
void save_final_config(io_context *ctx, system_state *s, double t, int NS);

//Hash & output directory helpers
char *read_file_to_string(const char *path);
unsigned long long compute_hash(const char *ini_path, int seed, const char *matrix_path);
void format_hash_hex(unsigned long long hash, char hex[17]);
int copy_file_to_dir(const char *src, const char *dir, const char *dst_name);
void write_metadata(const char *dir, int NS, int N, int D, int initial_type, double pth, double hm, double H, double t_final, const char *stop_reason);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Feedback
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void print_header_and_ini_parameters(const char *filename, int N, int D, int initial_type, int NS, double Amax, double phim, double dphi, double g, double beta, double hm, double dh, double Dh, double Dh2, double pth, double deq, double tmax, double dt, double t_coup, double t_eq, double t_pump, double t_pump2, int save_dyn, int new_coupling, double prob_coup, double prob_pm_re, double prob_pm_im, double coup_phase, int type_coup, double E_target, double E_target_tol, int stop_at_E_target, int stop_when_stable, double allowed_H_change, int steps_for_update, int stable_window, int relative_H_change, int buffer_size, int flush_interval, int num_threads, int save_steps, int compute_eigenvalues, int lattice_d, int lattice_NL, int lattice_bc, double t_hramp, int n_hramp, double Dt_hramp, double Dhramp) {

    printf("\n////////////////////////////////////////////////////////////////////////\n");
    printf("//                     COMPLEX HYPERSPIN MACHINE                      //\n");
    printf("//                                                                    //\n");
    printf("//   ^     ^  ^  ^        ^     ^        ^  ^     ^        ^          //\n");
    printf("//   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |    //\n");
    printf("//      v           v  v     v     v  v        v     v  v     v  v    //\n");
    printf("//                                                                    //\n");
    printf("//                                   last updated: 03/11/2026 - v2.9  //\n");
    printf("//                                               by T. Piacenti  :)   //\n");
    printf("////////////////////////////////////////////////////////////////////////\n\n");

    printf("-------------------------INITIALIZATION FILE----------------------------\n\n");
    printf("> '%s'\n\n",filename);

    printf("-------------------SIMULATION AND SYSTEM PARAMETERS---------------------\n\n");
    printf("> NS  = %d\n> N   = %d\n> D   = %d\n> Neq = %d\n\n",NS,N,D,2*N);

    printf("-------------------CAVITY PARAMETERS & EQUALIZATION---------------------\n\n");
    printf("> Amax = %0.4f\n> initial_type = %d\n> hm   = %0.4f\n> dh   = %0.4f\n> Dh   = %0.4f\n> Dh2  = %0.4f\n> phim = %0.4f\n> dphi = %0.4f\n> g    = %0.4f\n> beta = %0.4f\n> deq  = %0.4f\n\n",Amax,initial_type,hm,dh,Dh,Dh2,phim,dphi,g,beta,deq);

    printf("-------------------COUPLING MATRIX PARAMETERS---------------------------\n\n");
    if (new_coupling == 2){
        printf("> Lattice coupling matrix will be generated (new_coupling==2)\n");
        printf("> d-dimensional hypercubic lattice: d=%d, NL=%d, N=%d\n", lattice_d, lattice_NL, N);
        printf("> Boundary conditions: %s\n", lattice_bc ? "Periodic (PBC)" : "Open (OBC)");
        printf("> p_coup = %0.4f  (per-edge existence probability)\n", prob_coup);
        printf("> prob_pm_re = %0.4f\n> prob_pm_im = %0.4f\n> coup_phase = %0.4f\n\n", prob_pm_re, prob_pm_im, coup_phase);
    }
    else if (new_coupling == 1){
        printf("> New coupling matrix will be generated (new_coupling==1)\n");
        printf("> The new coupling matrix will be saved as 'Data_Matrix.dat'\n");
        printf("> p_coup = %0.4f\n> prob_pm_re = %0.4f\n> prob_pm_im = %0.4f\n> coup_phase = %0.4f\n\n",prob_coup,prob_pm_re,prob_pm_im,coup_phase);
    }
    else if (new_coupling == 0){
        printf("> Coupling matrix will be read from 'Data_Matrix.dat' (new_coupling==0)\n\n");
    }
    else {
        printf("> Unrecognized option for 'new_coupling' (must be equal to 0, 1, or 2): the program will likely not work.\n");
    }
    printf("> type_coup = %d  (0=real only, 1=imag only, 2=both Re+Im)\n", type_coup);
    if (compute_eigenvalues) {
        printf("> Eigenvalue computation enabled (Jmax, Lmax, h_threshold)\n");
        printf("> Dense coupling matrix allocated for eigenvalue diagnostics\n\n");
    } else {
        printf("> Eigenvalue computation disabled (Jmax/Lmax not computed)\n");
        printf("> Sparse-only mode: no dense coupling matrix allocated\n\n");
    }

    printf("-------------------------TIME PARAMETERS--------------------------------\n\n");
    printf("> tmax     = %0.2f\n> dt       = %0.4f\n> t_coup   = %0.2f\n> t_eq     = %0.2f\n> t_pump   = %0.2f\n> t_pump2  = %0.2f\n> t_hramp  = %0.2f\n> n_hramp  = %d\n> Dt_hramp = %0.4f\n> Dhramp   = %0.6f\n\n",tmax,dt,t_coup,t_eq,t_pump,t_pump2,t_hramp,n_hramp,Dt_hramp,Dhramp);

    printf("-------------------------PUMP THRESHOLD---------------------------------\n\n");
    if (pth < 0) {
        printf("> The value 'hm' will be used as average pump\n");
        printf("> pth = %0.3f\n\n",pth);
    }
    else if (pth >=0 ){
        printf("> The value 'hm' will be modified as: hm = pth*hth\n");
        printf("> pth = %0.3f\n",pth);
        printf("> hth to be computed...\n\n");
    }

    printf("--------------------------END CONDITIONS--------------------------------\n\n");
    if (stop_at_E_target == 1) {
        printf("> Simulation will stop when the system reaches the target energy.\n");
        printf("> If condition is not met will stop at tmax.\n");
        printf("> E_target     = %0.4f\n",E_target);
        printf("> E_target_tol = %0.12f\n",E_target_tol);
    } else if (stop_when_stable == 1) {
        printf("> Simulation will stop when the energy variations are below threshold.\n");
        printf("> Energy Variation checked every %d time steps (steps_for_update).\n",steps_for_update);
        printf("> Need %d consecutive stable checks (stable_window).\n",stable_window);
        printf("> If condition is not met will stop at tmax.\n");
        printf("> allowed_H_change = %0.12f\n",allowed_H_change);
        printf("> relative_H_change = %d  (%s)\n",relative_H_change,
               relative_H_change ? "|H-H_prev|/|H| < allowed_H_change" : "|H-H_prev| < allowed_H_change (absolute)");
        printf("\n");

    } else {
        printf("> Simulation will stop when 'tmax' is reached.\n\n");
    }

    printf("--------------------------OUTPUT FILES----------------------------------\n\n");
    if(save_dyn==0) {
        printf("> Fields state at tmax will be saved in 'Data_Final.dat'\n\n");
    } else if(save_dyn==1) {
        printf("> Fields state at tmax will be saved in 'Data_Final.dat'\n");
        printf("> All hyperspin dynamics saved in single merged file 'Data_Fields.dat'\n");
        printf("> Equalizers dynamics saved in 'Data_Equalizers.dat'\n");
        printf("> Energy dynamics saved in 'Data_Hamilton.dat'\n");
        printf("> Lyapunov dynamics saved in 'Data_Lyapunov.dat'\n");
    } else if(save_dyn==2) {
        printf("> Fields state at tmax will be saved in 'Data_Final.dat'\n");
        printf("> Energy dynamics saved in 'Data_Hamilton.dat'\n");
        printf("> Lyapunov dynamics saved in 'Data_Lyapunov.dat'\n");
    } else if(save_dyn==3) {
        printf("> Fields state at tmax will be saved in 'Data_Final.dat'\n");
        printf("> Energy, Lyapunov, Average Magnetization, Amplitude Variance in:\n");
        printf("  'Data_H_Ly_m_DS2.dat'\n\n");
    } else {
        printf("> Unrecognized option for 'save_dyn'.\n");
        printf("> Fields state at tmax will be saved in 'Data_Final.dat'\n\n");
    }

    printf("-------------------------BUFFERED I/O SETTINGS--------------------------\n\n");
    printf("> save_steps    = %d steps\n", save_steps);
    printf("> buffer_size   = %d bytes (%0.2f KB)\n", buffer_size, buffer_size/1024.0);
    printf("> flush_interval = %d steps\n\n", flush_interval);

    printf("-------------------------PARALLELIZATION--------------------------------\n\n");
    if (num_threads <= 1) {
        printf("> num_threads = %d (Sequential mode, OpenMP disabled)\n\n", num_threads);
    } else {
        printf("> num_threads = %d (Parallel mode, OpenMP enabled)\n\n", num_threads);
    }

    printf("------------------------------------------------------------------------\n");
    printf("------------------------------------------------------------------------\n\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Input
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int parse_config(const char *filename, int *N, int *D, int *initial_type, double *Amax, double *phim, double *dphi, double *g, double *beta, double *hm, double *dh, double *Dh, double *Dh2, double *deq, double *tmax, double *dt, double *t_coup, double *t_eq, double *t_pump, double *t_pump2, int *save_dyn, int *new_coupling, double *prob_coup, double *prob_pm_re, double *prob_pm_im, double *coup_phase, int *type_coup, double *E_target, double *E_target_tol, int *stop_at_E_target, int *stop_when_stable, double *allowed_H_change, int *steps_for_update, int *stable_window, int *relative_H_change, int *buffer_size, int *flush_interval, int *num_threads, int *save_steps, char *output_dir, size_t output_dir_sz, int *compute_eigenvalues, int *lattice_d, int *lattice_NL, int *lattice_bc, double *t_hramp, int *n_hramp, double *Dt_hramp, double *Dhramp) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open config file");
        return 1;
    }

    *buffer_size = 262144;
    *flush_interval = 1000;
    *num_threads = 1;
    *stable_window = 1000;
    *relative_H_change = 0;
    *steps_for_update = 1;
    *E_target_tol = 1e-10;
    *save_steps = 1;
    *compute_eigenvalues = 1;
    *lattice_d = 2;
    *lattice_NL = 10;
    *lattice_bc = 1;
    *initial_type = 2;
    *prob_pm_re = 0.5;
    *prob_pm_im = 0.5;
    *coup_phase = 0.0;
    *t_hramp = -1.0;
    *n_hramp = 0;
    *Dt_hramp = 0.0;
    *Dhramp = 0.0;
    *Dh2 = 0.0;
    *t_pump2 = -1.0;

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';

        char *key = line;
        char *value = eq + 1;
        while (*key == ' ' || *key == '\t') key++;
        int len = strlen(key);
        while (len > 0 && (key[len-1] == ' ' || key[len-1] == '\t' || key[len-1] == '\r' || key[len-1] == '\n')) key[--len] = '\0';

        char *end;
        while (*value == ' ' || *value == '\t') value++;
        len = strlen(value);
        while (len > 0 && (value[len-1] == ' ' || value[len-1] == '\t' || value[len-1] == '\r' || value[len-1] == '\n')) value[--len] = '\0';

        if (strcmp(key, "N") == 0) *N = atoi(value);
        else if (strcmp(key, "D") == 0) *D = atoi(value);
        else if (strcmp(key, "Amax") == 0) *Amax = strtod(value, &end);
        else if (strcmp(key, "initial_type") == 0) *initial_type = atoi(value);
        else if (strcmp(key, "phim") == 0) *phim = strtod(value, &end);
        else if (strcmp(key, "dphi") == 0) *dphi = strtod(value, &end);
        else if (strcmp(key, "gm")==0) *g = strtod(value, &end);
        else if (strcmp(key, "beta")==0) *beta = strtod(value, &end);
        else if (strcmp(key, "hm")==0) *hm = strtod(value, &end);
        else if (strcmp(key, "dh")==0) *dh = strtod(value, &end);
        else if (strcmp(key, "Dh")==0) *Dh = strtod(value, &end);
        else if (strcmp(key, "Dh2")==0) *Dh2 = strtod(value, &end);
        else if (strcmp(key, "deq")==0) *deq = strtod(value, &end);
        else if (strcmp(key, "tmax")==0) *tmax = strtod(value, &end);
        else if (strcmp(key, "dt")==0) *dt = strtod(value, &end);
        else if (strcmp(key, "t_coup")==0) *t_coup = strtod(value, &end);
        else if (strcmp(key, "t_eq")==0) *t_eq = strtod(value, &end);
        else if (strcmp(key, "t_pump")==0) *t_pump = strtod(value, &end);
        else if (strcmp(key, "t_pump2")==0) *t_pump2 = strtod(value, &end);
        else if (strcmp(key, "t_hramp")==0)  *t_hramp = strtod(value, &end);
        else if (strcmp(key, "n_hramp")==0)  *n_hramp = atoi(value);
        else if (strcmp(key, "Dt_hramp")==0) *Dt_hramp = strtod(value, &end);
        else if (strcmp(key, "Dhramp")==0)   *Dhramp = strtod(value, &end);
        else if (strcmp(key, "save_dyn")==0) *save_dyn = atoi(value);
        else if (strcmp(key, "new_coupling")==0) *new_coupling = atoi(value);
        else if (strcmp(key, "prob_coup")==0) *prob_coup = strtod(value, &end);
        else if (strcmp(key, "prob_pm_re")==0) *prob_pm_re = strtod(value, &end);
        else if (strcmp(key, "prob_pm_im")==0) *prob_pm_im = strtod(value, &end);
        else if (strcmp(key, "coup_phase")==0) *coup_phase = strtod(value, &end);
        else if (strcmp(key, "type_coup") == 0) *type_coup = atoi(value);
        else if (strcmp(key, "E_target")==0) *E_target = strtod(value, &end);
        else if (strcmp(key, "E_target_tol")==0) *E_target_tol = strtod(value, &end);
        else if (strcmp(key, "stop_at_E_target") == 0) *stop_at_E_target = atoi(value);
        else if (strcmp(key, "stop_when_stable") == 0) *stop_when_stable = atoi(value);
        else if (strcmp(key, "allowed_H_change")==0) *allowed_H_change = strtod(value, &end);
        else if (strcmp(key, "steps_for_update") == 0) *steps_for_update = atoi(value);
        else if (strcmp(key, "stable_window") == 0) *stable_window = atoi(value);
        else if (strcmp(key, "relative_H_change") == 0) *relative_H_change = atoi(value);
        else if (strcmp(key, "buffer_size") == 0) *buffer_size = atoi(value);
        else if (strcmp(key, "flush_interval") == 0) *flush_interval = atoi(value);
        else if (strcmp(key, "num_threads") == 0) *num_threads = atoi(value);
        else if (strcmp(key, "save_steps") == 0) *save_steps = atoi(value);
        else if (strcmp(key, "output_dir") == 0) {
            strncpy(output_dir, value, output_dir_sz - 1);
            output_dir[output_dir_sz - 1] = '\0';
        }
        else if (strcmp(key, "compute_eigenvalues") == 0) *compute_eigenvalues = atoi(value);
        else if (strcmp(key, "lattice_d") == 0) *lattice_d = atoi(value);
        else if (strcmp(key, "lattice_NL") == 0) *lattice_NL = atoi(value);
        else if (strcmp(key, "lattice_bc") == 0) *lattice_bc = atoi(value);
        else {
            fprintf(stderr, "Unknown key: %s\n", key);
        }
    }
    fclose(file);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//System State Management
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int allocate_system(int N, int D, system_state *s) {
    s->N = N;
    s->D = D;

    s->S_re   = calloc(N*D, sizeof(double));
    s->S_im   = calloc(N*D, sizeof(double));
    s->S2     = calloc(N,   sizeof(double));
    s->DS_re  = calloc(N*D, sizeof(double));
    s->DS_im  = calloc(N*D, sizeof(double));
    s->DS2    = calloc(N,   sizeof(double));
    s->Y_re   = calloc(N*2, sizeof(double));
    s->Y_im   = calloc(N*2, sizeof(double));
    s->DY_re  = calloc(N*2, sizeof(double));
    s->DY_im  = calloc(N*2, sizeof(double));
    s->J_re   = NULL;
    s->J_im   = NULL;
    s->J_allocated = 0;
    s->h      = calloc(N,   sizeof(double));
    s->hmod   = calloc(N,   sizeof(double));
    s->phi    = calloc(N,   sizeof(double));
    s->JS_re  = calloc(N*D, sizeof(double));
    s->JS_im  = calloc(N*D, sizeof(double));
    // Quick-win scratch arrays
    s->cos_phi    = calloc(N, sizeof(double));
    s->sin_phi    = calloc(N, sizeof(double));
    s->inv_sqrtS2 = calloc(N, sizeof(double));

    if (!s->S_re || !s->S_im || !s->S2 || !s->DS_re || !s->DS_im ||
        !s->DS2 || !s->Y_re || !s->Y_im || !s->DY_re || !s->DY_im ||
        !s->h || !s->hmod || !s->phi ||
        !s->JS_re || !s->JS_im || !s->cos_phi || !s->sin_phi || !s->inv_sqrtS2) {
        return 1;
    }
    return 0;
}

void free_system(system_state *s) {
    free(s->S_re);  free(s->S_im);  free(s->S2);
    free(s->DS_re); free(s->DS_im); free(s->DS2);
    free(s->Y_re);  free(s->Y_im);
    free(s->DY_re); free(s->DY_im);
    if (s->J_allocated) {
        free(s->J_re);
        free(s->J_im);
    }
    free(s->h);     free(s->hmod);  free(s->phi);
    free(s->JS_re); free(s->JS_im);
    free(s->cos_phi); free(s->sin_phi); free(s->inv_sqrtS2);
    free(s->csr_row_ptr); free(s->csr_col_idx);
    free(s->csr_Jr);      free(s->csr_Ji);
    free(s->edge_a);      free(s->edge_b);
    free(s->edge_Jr);     free(s->edge_Ji);
    memset(s, 0, sizeof(*s));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Initialization
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_fields(system_state *s, double Amax, int initial_type) {
    int N = s->N, D = s->D;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            S_RE(s, i, j) = -Amax + (2.0 * Amax * rand() / (RAND_MAX + 1.0));
            S_IM(s, i, j) = -Amax + (2.0 * Amax * rand() / (RAND_MAX + 1.0));
            if (initial_type == 0) S_IM(s, i, j) = 0.0;
            else if (initial_type == 1) S_RE(s, i, j) = 0.0;
        }
        for (int j = 0; j < 2; j++) {
            Y_RE(s, i, j) = -Amax + (2.0 * Amax * rand() / (RAND_MAX + 1.0));
            Y_IM(s, i, j) = -Amax + (2.0 * Amax * rand() / (RAND_MAX + 1.0));
            if (initial_type == 0) Y_IM(s, i, j) = 0.0;
            else if (initial_type == 1) Y_RE(s, i, j) = 0.0;
        }
    }
}

void initialize_pump(system_state *s, double *target, double hm, double dh) {
    for (int i = 0; i < s->N; i++) {
        double a = -dh + (2.0 * dh * rand() / (RAND_MAX + 1.0));
        target[i] = hm + a;
    }
}

void modify_pump(system_state *s) {
    for (int i = 0; i < s->N; i++) {
        s->h[i] += s->hmod[i];
    }
}

void ramp_pump(system_state *s, double Dhramp) {
    for (int i = 0; i < s->N; i++) {
        s->h[i] += Dhramp;
    }
}

void precompute_trig(system_state *s) {
    for (int i = 0; i < s->N; i++) {
        s->cos_phi[i] = cos(s->phi[i]);
        s->sin_phi[i] = sin(s->phi[i]);
    }
}

void initialize_J_Hermitian_Binary_Random(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup) {
    int N = s->N;
    double dJR, dJI;

    if (type_coup == 0) {
        dJR = 1.0 / (N * prob_coup);
        dJI = 0;
    } else if (type_coup == 1) {
        dJR = 0;
        dJI = 1.0 / (N * prob_coup);
    } else {
        dJR = 1.0 / (sqrt(2.0) * N * prob_coup);
        dJI = 1.0 / (sqrt(2.0) * N * prob_coup);
    }

    for (int i = 0; i < N; i++) {
        J_RE(s, i, i) = 0;
        J_IM(s, i, i) = 0;
        for (int j = 0; j < i; j++) {
            int coup_yn = ((double)rand() / (double)RAND_MAX) < prob_coup;
            if (coup_yn == 0) {
                J_RE(s, i, j) = 0; J_IM(s, i, j) = 0;
                J_RE(s, j, i) = 0; J_IM(s, j, i) = 0;
            } else {
                int coup_pm_re = ((double)rand() / (double)RAND_MAX) < prob_pm_re;
                int coup_pm_im = ((double)rand() / (double)RAND_MAX) < prob_pm_im;
                double sign_re = -1.0 + coup_pm_re * 2.0;
                double sign_im = -1.0 + coup_pm_im * 2.0;
                double J1 = dJR * sign_re;
                double J2 = -dJI * sign_im;

                J_RE(s, i, j) = J1;  J_IM(s, i, j) = -J2;
                J_RE(s, j, i) = J1;  J_IM(s, j, i) = J2;
            }
        }
    }

    // Save coupling matrix
    FILE *fp = fopen("Data_Matrix.dat", "w");
    if (!fp) { fprintf(stderr, "Error creating Data_Matrix.dat\n"); return; }
    fprintf(fp, "\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%0.15f  %0.15f    ", J_RE(s, i, j), J_IM(s, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void initialize_J_Hermitian_Binary_Sparse(system_state *s, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup, int NS) {
    int N = s->N;
    double dJR, dJI;

    if (type_coup == 0) {
        dJR = 1.0 / (N * prob_coup);
        dJI = 0;
    } else if (type_coup == 1) {
        dJR = 0;
        dJI = 1.0 / (N * prob_coup);
    } else {
        dJR = 1.0 / (sqrt(2.0) * N * prob_coup);
        dJI = 1.0 / (sqrt(2.0) * N * prob_coup);
    }

    // Pass 1: count per-row nnz via reseeded rand_r (no pair_data storage)
    unsigned int seed = (unsigned int)NS;
    int nnz = 0;
    int *row_count = calloc(N, sizeof(int));
    if (!row_count) { fprintf(stderr, "Memory allocation failed in initialize_J_Hermitian_Binary_Sparse\n"); return; }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            int exists = ((double)rand_r(&seed) / (double)RAND_MAX) < prob_coup;
            if (exists) {
                (void)((double)rand_r(&seed) / (double)RAND_MAX);
                (void)((double)rand_r(&seed) / (double)RAND_MAX);
                row_count[i]++;
                row_count[j]++;
                nnz += 2;
            }
        }
    }

    // Edge count
    s->num_edges = nnz / 2;

    // Allocate sparse structures
    s->csr_row_ptr = calloc(N + 1, sizeof(int));
    s->csr_col_idx = calloc(nnz, sizeof(int));
    s->csr_Jr      = calloc(nnz, sizeof(double));
    s->csr_Ji      = calloc(nnz, sizeof(double));
    s->edge_a      = calloc(s->num_edges, sizeof(int));
    s->edge_b      = calloc(s->num_edges, sizeof(int));
    s->edge_Jr     = calloc(s->num_edges, sizeof(double));
    s->edge_Ji     = calloc(s->num_edges, sizeof(double));
    if (!s->csr_row_ptr || !s->csr_col_idx || !s->csr_Jr || !s->csr_Ji ||
        !s->edge_a || !s->edge_b || !s->edge_Jr || !s->edge_Ji) {
        fprintf(stderr, "Memory allocation failed in initialize_J_Hermitian_Binary_Sparse\n");
        free(row_count); return;
    }

    // Build row_ptr from row_count
    for (int i = 0; i < N; i++) {
        s->csr_row_ptr[i + 1] = s->csr_row_ptr[i] + row_count[i];
    }

    // Pass 2: reseed and fill CSR + edge list
    seed = (unsigned int)NS;
    int *cur_row = calloc(N, sizeof(int));
    if (!cur_row) { free(row_count); return; }
    for (int i = 0; i < N; i++) cur_row[i] = s->csr_row_ptr[i];

    int edge_idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            int exists = ((double)rand_r(&seed) / (double)RAND_MAX) < prob_coup;
            if (exists) {
                int sign_re = ((double)rand_r(&seed) / (double)RAND_MAX) < prob_pm_re;
                int sign_im = ((double)rand_r(&seed) / (double)RAND_MAX) < prob_pm_im;
                double coup_sign_re = -1.0 + (double)sign_re * 2.0;
                double coup_sign_im = -1.0 + (double)sign_im * 2.0;
                double J1 = dJR * coup_sign_re;
                double J2 = -dJI * coup_sign_im;

                // Row i -> col j: J(i,j) = J1 + i*(-J2) = dJR*sign + i*dJI*sign
                int ki = cur_row[i]++;
                s->csr_col_idx[ki] = j;
                s->csr_Jr[ki] = J1;
                s->csr_Ji[ki] = -J2;

                // Row j -> col i: J(j,i) = J1 + i*J2 = dJR*sign - i*dJI*sign
                int kj = cur_row[j]++;
                s->csr_col_idx[kj] = i;
                s->csr_Jr[kj] = J1;
                s->csr_Ji[kj] = J2;

                // Edge list (undirected, smaller index first)
                s->edge_a[edge_idx] = j;
                s->edge_b[edge_idx] = i;
                s->edge_Jr[edge_idx] = J1;
                s->edge_Ji[edge_idx] = J2;
                edge_idx++;
            }
        }
    }

    // Write sparse matrix file
    write_sparse_matrix_file(s, "Data_Matrix_sparse.dat");

    free(row_count);
    free(cur_row);
    struct mallinfo2 mi_startup = mallinfo2();
    s->startup_heap_B = (unsigned long)(mi_startup.uordblks + mi_startup.hblkhd);
}

void initialize_J_Read_From_File(system_state *s, const char *filename) {
    int N = s->N;
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error reading %s\n", filename); return; }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(fp, "%lf %lf", &J_RE(s, i, j), &J_IM(s, i, j)) != 2) {
                fprintf(stderr, "Error reading element (%d,%d) from %s\n", i, j, filename);
                fclose(fp);
                return;
            }
        }
    }
    fclose(fp);
}

void build_sparse_coupling(system_state *s) {
    int N = s->N;
    int nnz = 0;
    int *row_count = calloc(N, sizeof(int));
    if (!row_count) { fprintf(stderr, "Memory allocation failed in build_sparse_coupling\n"); return; }

    // Pass 1: count non-zero off-diagonals per row
    for (int a = 0; a < N; a++) {
        int cnt = 0;
        for (int b = 0; b < N; b++) {
            if (a == b) continue;
            if (J_RE(s, a, b) != 0.0 || J_IM(s, a, b) != 0.0) cnt++;
        }
        row_count[a] = cnt;
        nnz += cnt;
    }

    s->num_edges = nnz / 2;
    s->csr_row_ptr = calloc(N + 1, sizeof(int));
    s->csr_row_ptr[0] = 0;
    for (int a = 0; a < N; a++) {
        s->csr_row_ptr[a + 1] = s->csr_row_ptr[a] + row_count[a];
    }

    s->csr_col_idx = calloc(nnz, sizeof(int));
    s->csr_Jr      = calloc(nnz, sizeof(double));
    s->csr_Ji      = calloc(nnz, sizeof(double));
    s->edge_a      = calloc(s->num_edges, sizeof(int));
    s->edge_b      = calloc(s->num_edges, sizeof(int));
    s->edge_Jr     = calloc(s->num_edges, sizeof(double));
    s->edge_Ji     = calloc(s->num_edges, sizeof(double));
    if (!s->csr_row_ptr || !s->csr_col_idx || !s->csr_Jr || !s->csr_Ji ||
        !s->edge_a || !s->edge_b || !s->edge_Jr || !s->edge_Ji) {
        fprintf(stderr, "Memory allocation failed in build_sparse_coupling\n");
        free(row_count); return;
    }

    // Pass 2: fill CSR and edge list
    int *cur_row = calloc(N, sizeof(int));
    if (!cur_row) { free(row_count); return; }
    for (int a = 0; a < N; a++) cur_row[a] = s->csr_row_ptr[a];

    int edge_idx = 0;
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {
            if (a == b) continue;
            if (J_RE(s, a, b) != 0.0 || J_IM(s, a, b) != 0.0) {
                int k = cur_row[a]++;
                s->csr_col_idx[k] = b;
                s->csr_Jr[k] = J_RE(s, a, b);
                s->csr_Ji[k] = J_IM(s, a, b);
                if (a < b) {
                    s->edge_a[edge_idx] = a;
                    s->edge_b[edge_idx] = b;
                    s->edge_Jr[edge_idx] = J_RE(s, a, b);
                    s->edge_Ji[edge_idx] = J_IM(s, a, b);
                    edge_idx++;
                }
            }
        }
    }

    free(row_count);
    free(cur_row);
}

void write_sparse_matrix_file(system_state *s, const char *filename) {
    if (!s->edge_a) { fprintf(stderr, "Error: edge list not available for sparse file write\n"); return; }
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error creating %s\n", filename); return; }
    fprintf(fp, "%d\n", s->num_edges);
    for (int e = 0; e < s->num_edges; e++)
        fprintf(fp, "%d  %d  %0.15f  %0.15f\n", s->edge_a[e], s->edge_b[e], s->edge_Jr[e], s->edge_Ji[e]);
    fclose(fp);
}

void write_dense_matrix_file(system_state *s, const char *filename) {
    int N = s->N;
    if (!s->J_re || !s->J_im) { fprintf(stderr, "Error: dense J not available for dense file write\n"); return; }
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Error creating %s\n", filename); return; }
    fprintf(fp, "\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            fprintf(fp, "%0.15f  %0.15f    ", J_RE(s, i, j), J_IM(s, i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void fill_dense_from_sparse(system_state *s) {
    int N = s->N;
    memset(s->J_re, 0, (size_t)N * N * sizeof(double));
    memset(s->J_im, 0, (size_t)N * N * sizeof(double));
    for (int e = 0; e < s->num_edges; e++) {
        int a = s->edge_a[e], b = s->edge_b[e];
        double Jr = s->edge_Jr[e], Ji = s->edge_Ji[e];
        J_RE(s, a, b) = Jr;
        J_IM(s, a, b) = Ji;
        J_RE(s, b, a) = Jr;
        J_IM(s, b, a) = -Ji;
    }
}



void initialize_J_Sparse_From_File(system_state *s, const char *filename) {
    int N = s->N;

    // Pass 1: read all entries, count non-zero off-diagonals per row
    int nnz = 0;
    int *row_count = calloc(N, sizeof(int));
    if (!row_count) { fprintf(stderr, "Memory allocation failed in initialize_J_Sparse_From_File\n"); return; }

    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error reading %s\n", filename); free(row_count); return; }
    for (int i = 0; i < N; i++) {
        int cnt = 0;
        for (int j = 0; j < N; j++) {
            double re, im;
            if (fscanf(fp, "%lf %lf", &re, &im) != 2) {
                fprintf(stderr, "Error reading element (%d,%d) from %s\n", i, j, filename);
                fclose(fp); free(row_count); return;
            }
            if (i != j && (re != 0.0 || im != 0.0)) cnt++;
        }
        row_count[i] = cnt;
        nnz += cnt;
    }
    fclose(fp);

    s->num_edges = nnz / 2;
    s->csr_row_ptr = calloc(N + 1, sizeof(int));
    s->csr_col_idx = calloc(nnz, sizeof(int));
    s->csr_Jr      = calloc(nnz, sizeof(double));
    s->csr_Ji      = calloc(nnz, sizeof(double));
    s->edge_a      = calloc(s->num_edges, sizeof(int));
    s->edge_b      = calloc(s->num_edges, sizeof(int));
    s->edge_Jr     = calloc(s->num_edges, sizeof(double));
    s->edge_Ji     = calloc(s->num_edges, sizeof(double));
    if (!s->csr_row_ptr || !s->csr_col_idx || !s->csr_Jr || !s->csr_Ji ||
        !s->edge_a || !s->edge_b || !s->edge_Jr || !s->edge_Ji) {
        fprintf(stderr, "Memory allocation failed in initialize_J_Sparse_From_File\n");
        free(row_count); return;
    }

    // Build row_ptr
    for (int i = 0; i < N; i++) {
        s->csr_row_ptr[i + 1] = s->csr_row_ptr[i] + row_count[i];
    }

    // Pass 2: rewind and fill CSR + edge list
    fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error re-reading %s\n", filename); free(row_count); return; }
    int *cur_row = calloc(N, sizeof(int));
    if (!cur_row) { free(row_count); fclose(fp); return; }
    for (int i = 0; i < N; i++) cur_row[i] = s->csr_row_ptr[i];

    int edge_idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double re, im;
            if (fscanf(fp, "%lf %lf", &re, &im) != 2) {
                fprintf(stderr, "Error re-reading element (%d,%d)\n", i, j);
                free(row_count); free(cur_row); fclose(fp); return;
            }
            if (i != j && (re != 0.0 || im != 0.0)) {
                int k = cur_row[i]++;
                s->csr_col_idx[k] = j;
                s->csr_Jr[k] = re;
                s->csr_Ji[k] = im;
                if (i < j) {
                    s->edge_a[edge_idx] = i;
                    s->edge_b[edge_idx] = j;
                    s->edge_Jr[edge_idx] = re;
                    s->edge_Ji[edge_idx] = im;
                    edge_idx++;
                }
            }
        }
    }
    fclose(fp);
    free(row_count);
    free(cur_row);
}

void initialize_J_Sparse_From_Sparse_File(system_state *s, const char *filename) {
    int N = s->N;
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error reading sparse file %s\n", filename); return; }

    int num_edges;
    if (fscanf(fp, "%d", &num_edges) != 1) {
        fprintf(stderr, "Error reading num_edges from %s\n", filename);
        fclose(fp); return;
    }

    // Pass 1: read all edges, count per-row degree for CSR
    int *row_count = calloc(N, sizeof(int));
    if (!row_count) { fclose(fp); fprintf(stderr, "Memory allocation failed\n"); return; }
    for (int e = 0; e < num_edges; e++) {
        int a, b; double Jr, Ji;
        if (fscanf(fp, "%d %d %lf %lf", &a, &b, &Jr, &Ji) != 4) {
            fprintf(stderr, "Error reading edge %d from %s\n", e, filename);
            free(row_count); fclose(fp); return;
        }
        row_count[a]++;
        row_count[b]++;
    }
    fclose(fp);

    int nnz = 0;
    for (int i = 0; i < N; i++) nnz += row_count[i];

    // Allocate sparse structures
    s->num_edges = num_edges;
    s->csr_row_ptr = calloc(N + 1, sizeof(int));
    s->csr_col_idx = calloc(nnz, sizeof(int));
    s->csr_Jr      = calloc(nnz, sizeof(double));
    s->csr_Ji      = calloc(nnz, sizeof(double));
    s->edge_a      = calloc(num_edges, sizeof(int));
    s->edge_b      = calloc(num_edges, sizeof(int));
    s->edge_Jr     = calloc(num_edges, sizeof(double));
    s->edge_Ji     = calloc(num_edges, sizeof(double));
    if (!s->csr_row_ptr || !s->csr_col_idx || !s->csr_Jr || !s->csr_Ji ||
        !s->edge_a || !s->edge_b || !s->edge_Jr || !s->edge_Ji) {
        fprintf(stderr, "Memory allocation failed in initialize_J_Sparse_From_Sparse_File\n");
        free(row_count); return;
    }

    // Build row_ptr from row_count
    for (int i = 0; i < N; i++) {
        s->csr_row_ptr[i + 1] = s->csr_row_ptr[i] + row_count[i];
    }

    // Pass 2: rewind and fill CSR + edge list
    fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error re-reading %s\n", filename); free(row_count); return; }
    int dummy;
    if (fscanf(fp, "%d", &dummy) != 1) { fclose(fp); free(row_count); return; }

    int *cur_row = calloc(N, sizeof(int));
    if (!cur_row) { free(row_count); fclose(fp); return; }
    for (int i = 0; i < N; i++) cur_row[i] = s->csr_row_ptr[i];

    for (int e = 0; e < num_edges; e++) {
        int a, b; double Jr, Ji;
        if (fscanf(fp, "%d %d %lf %lf", &a, &b, &Jr, &Ji) != 4) {
            fprintf(stderr, "Error re-reading edge %d\n", e);
            free(row_count); free(cur_row); fclose(fp); return;
        }
        // File stores (a,b) with a < b: Jr + i*Ji = J(a,b) (upper triangle)
        // CSR row a -> col b: upper triangle value
        int ka = cur_row[a]++;
        s->csr_col_idx[ka] = b;
        s->csr_Jr[ka] = Jr;
        s->csr_Ji[ka] = Ji;
        // CSR row b -> col a: lower triangle (Hermitian conjugate)
        int kb = cur_row[b]++;
        s->csr_col_idx[kb] = a;
        s->csr_Jr[kb] = Jr;
        s->csr_Ji[kb] = -Ji;
        // Edge list: store as-read with original ordering & values
        s->edge_a[e] = a;
        s->edge_b[e] = b;
        s->edge_Jr[e] = Jr;
        s->edge_Ji[e] = Ji;
    }
    fclose(fp);
    free(row_count);
    free(cur_row);
}

// Coordinate conversion helpers for d-dimensional hypercube
// Row-major: last dimension varies fastest (C ordering)
static inline void index_to_coord(int idx, int NL, int d, int *coord) {
    for (int dim = 0; dim < d; dim++) {
        coord[dim] = idx % NL;
        idx /= NL;
    }
}
static inline int coord_to_index(const int *coord, int NL, int d) {
    int idx = 0;
    int stride = 1;
    for (int dim = 0; dim < d; dim++) {
        idx += coord[dim] * stride;
        stride *= NL;
    }
    return idx;
}

void initialize_J_Hermitian_Lattice(system_state *s, int lattice_d, int lattice_NL, int lattice_bc, double prob_pm_re, double prob_pm_im, double prob_coup, int type_coup) {
    int N = s->N;
    int degree = 2 * lattice_d;

    // Normalization: same structure as random case, 1/(expected_degree*prob_coup)
    double dJR, dJI;
    if (type_coup == 0) {
        dJR = 1.0 / (degree * prob_coup);
        dJI = 0;
    } else if (type_coup == 1) {
        dJR = 0;
        dJI = 1.0 / (degree * prob_coup);
    } else {
        dJR = 1.0 / (sqrt(2.0) * degree * prob_coup);
        dJI = 1.0 / (sqrt(2.0) * degree * prob_coup);
    }

    // Validate N == NL^d
    int expected_N = 1;
    for (int i = 0; i < lattice_d; i++) expected_N *= lattice_NL;
    if (N != expected_N) {
        fprintf(stderr, "Error: N=%d but NL^d = %d^%d = %d\n", N, lattice_NL, lattice_d, expected_N);
        return;
    }

    // Two-pass construction:
    // Pass 1: for each spin a, examine each dimension in both directions (forward/backward).
    // When a neighbor b > a is found (upper triangle), generate random edge decision
    // and count CSR row entries. Store decisions in a packed list for pass 2.
    int max_decisions = N * lattice_d * 2;  // forward + backward per (a, dim)
    struct { int b; int exists; int sign_re; int sign_im; } *edge_decisions = malloc(max_decisions * sizeof(*edge_decisions));
    if (!edge_decisions) { fprintf(stderr, "Memory allocation failed\n"); return; }
    int decision_count = 0;

    int total_edges = 0;
    int *row_count = calloc(N, sizeof(int));
    if (!row_count) { free(edge_decisions); fprintf(stderr, "Memory allocation failed\n"); return; }

    int coord[16];  // d <= 16
    for (int a = 0; a < N; a++) {
        index_to_coord(a, lattice_NL, lattice_d, coord);
        for (int dim = 0; dim < lattice_d; dim++) {
            int bf, bb;
            int fwd_valid = 1, bwd_valid = 1;

            // Forward direction
            int bf_coord[16];
            memcpy(bf_coord, coord, lattice_d * sizeof(int));
            bf_coord[dim]++;
            if (bf_coord[dim] >= lattice_NL) {
                if (lattice_bc) bf_coord[dim] = 0;
                else fwd_valid = 0;
            }
            bf = fwd_valid ? coord_to_index(bf_coord, lattice_NL, lattice_d) : -1;

            // Backward direction
            int bb_coord[16];
            memcpy(bb_coord, coord, lattice_d * sizeof(int));
            bb_coord[dim]--;
            if (bb_coord[dim] < 0) {
                if (lattice_bc) bb_coord[dim] = lattice_NL - 1;
                else bwd_valid = 0;
            }
            bb = bwd_valid ? coord_to_index(bb_coord, lattice_NL, lattice_d) : -1;

            // Store forward decision (always, even if bf <= a)
            int fwd_exists = 0, fwd_sign_re = 0, fwd_sign_im = 0;
            if (fwd_valid && bf > a) {
                fwd_exists = ((double)rand() / (double)RAND_MAX) < prob_coup;
                if (fwd_exists) {
                    fwd_sign_re = ((double)rand() / (double)RAND_MAX) < prob_pm_re;
                    fwd_sign_im = ((double)rand() / (double)RAND_MAX) < prob_pm_im;
                    row_count[a]++; row_count[bf]++; total_edges++;
                }
            }
            edge_decisions[decision_count].b = bf;
            edge_decisions[decision_count].exists = fwd_exists;
            edge_decisions[decision_count].sign_re = fwd_sign_re;
            edge_decisions[decision_count].sign_im = fwd_sign_im;
            decision_count++;

            // Store backward decision (always, even if bb <= a)
            int bwd_exists = 0, bwd_sign_re = 0, bwd_sign_im = 0;
            if (bwd_valid && bb > a) {
                bwd_exists = ((double)rand() / (double)RAND_MAX) < prob_coup;
                if (bwd_exists) {
                    bwd_sign_re = ((double)rand() / (double)RAND_MAX) < prob_pm_re;
                    bwd_sign_im = ((double)rand() / (double)RAND_MAX) < prob_pm_im;
                    row_count[a]++; row_count[bb]++; total_edges++;
                }
            }
            edge_decisions[decision_count].b = bb;
            edge_decisions[decision_count].exists = bwd_exists;
            edge_decisions[decision_count].sign_re = bwd_sign_re;
            edge_decisions[decision_count].sign_im = bwd_sign_im;
            decision_count++;
        }
    }

    // Build CSR prefix sum
    s->csr_row_ptr = malloc((N + 1) * sizeof(int));
    s->csr_row_ptr[0] = 0;
    for (int a = 0; a < N; a++) {
        s->csr_row_ptr[a + 1] = s->csr_row_ptr[a] + row_count[a];
    }
    int nnz = s->csr_row_ptr[N];

    s->num_edges = total_edges;
    s->csr_col_idx = malloc(nnz * sizeof(int));
    s->csr_Jr      = calloc(nnz, sizeof(double));
    s->csr_Ji      = calloc(nnz, sizeof(double));
    s->edge_a      = malloc(total_edges * sizeof(int));
    s->edge_b      = malloc(total_edges * sizeof(int));
    s->edge_Jr     = malloc(total_edges * sizeof(double));
    s->edge_Ji     = malloc(total_edges * sizeof(double));
    if (!s->csr_row_ptr || !s->csr_col_idx || !s->csr_Jr || !s->csr_Ji ||
        !s->edge_a || !s->edge_b || !s->edge_Jr || !s->edge_Ji) {
        fprintf(stderr, "Memory allocation failed\n");
        free(edge_decisions); free(row_count); return;
    }

    // Fill dense J if allocated
    if (s->J_allocated) {
        memset(s->J_re, 0, N * N * sizeof(double));
        memset(s->J_im, 0, N * N * sizeof(double));
    }

    // Pass 2: fill CSR and edge list from stored decisions
    int *cur_row = calloc(N, sizeof(int));
    if (!cur_row) { free(edge_decisions); free(row_count); return; }
    for (int a = 0; a < N; a++) cur_row[a] = s->csr_row_ptr[a];

    int edge_idx = 0;
    int dec_idx = 0;
    for (int a = 0; a < N; a++) {
        for (int dim = 0; dim < lattice_d; dim++) {
            // Forward direction
            if (edge_decisions[dec_idx].exists) {
                int b = edge_decisions[dec_idx].b;
                double sign_re = -1.0 + edge_decisions[dec_idx].sign_re * 2.0;
                double sign_im = -1.0 + edge_decisions[dec_idx].sign_im * 2.0;
                double J1 = dJR * sign_re;
                double J2 = dJI * sign_im;
                int ka = cur_row[a]++;
                s->csr_col_idx[ka] = b;
                s->csr_Jr[ka] = J1;
                s->csr_Ji[ka] = -J2;
                int kb = cur_row[b]++;
                s->csr_col_idx[kb] = a;
                s->csr_Jr[kb] = J1;
                s->csr_Ji[kb] = J2;
                s->edge_a[edge_idx] = a;
                s->edge_b[edge_idx] = b;
                s->edge_Jr[edge_idx] = J1;
                s->edge_Ji[edge_idx] = -J2;
                edge_idx++;
                if (s->J_allocated) {
                    J_RE(s, a, b) = J1;
                    J_IM(s, a, b) = -J2;
                    J_RE(s, b, a) = J1;
                    J_IM(s, b, a) = J2;
                }
            }
            dec_idx++;

            // Backward direction
            if (edge_decisions[dec_idx].exists) {
                int b = edge_decisions[dec_idx].b;
                double sign_re = -1.0 + edge_decisions[dec_idx].sign_re * 2.0;
                double sign_im = -1.0 + edge_decisions[dec_idx].sign_im * 2.0;
                double J1 = dJR * sign_re;
                double J2 = dJI * sign_im;
                int ka = cur_row[a]++;
                s->csr_col_idx[ka] = b;
                s->csr_Jr[ka] = J1;
                s->csr_Ji[ka] = -J2;
                int kb = cur_row[b]++;
                s->csr_col_idx[kb] = a;
                s->csr_Jr[kb] = J1;
                s->csr_Ji[kb] = J2;
                s->edge_a[edge_idx] = a;
                s->edge_b[edge_idx] = b;
                s->edge_Jr[edge_idx] = J1;
                s->edge_Ji[edge_idx] = -J2;
                edge_idx++;
                if (s->J_allocated) {
                    J_RE(s, a, b) = J1;
                    J_IM(s, a, b) = -J2;
                    J_RE(s, b, a) = J1;
                    J_IM(s, b, a) = J2;
                }
            }
            dec_idx++;
        }
    }

    // Write sparse matrix file
    write_sparse_matrix_file(s, "Data_Matrix_sparse.dat");

    // Write dense matrix file if dense J is available
    if (s->J_allocated) {
        write_dense_matrix_file(s, "Data_Matrix.dat");
    }

    free(edge_decisions);
    free(row_count);
    free(cur_row);
}

void apply_coupling_rotation(system_state *s, double coup_phase) {
    if (coup_phase == 0.0) return;
    double rot_cos = cos(coup_phase);
    double rot_sin = sin(coup_phase);
    for (int e = 0; e < s->num_edges; e++) {
        int a = s->edge_a[e];
        int b = s->edge_b[e];   // a < b by construction
        double old_Jr = s->edge_Jr[e];
        double old_Ji = s->edge_Ji[e];
        double new_Jr = old_Jr * rot_cos - old_Ji * rot_sin;
        double new_Ji = old_Jr * rot_sin + old_Ji * rot_cos;
        // Edge list (upper triangle)
        s->edge_Jr[e] = new_Jr;
        s->edge_Ji[e] = new_Ji;
        // CSR upper: row a -> col b
        for (int k = s->csr_row_ptr[a]; k < s->csr_row_ptr[a+1]; k++) {
            if (s->csr_col_idx[k] == b) {
                s->csr_Jr[k] = new_Jr;
                s->csr_Ji[k] = new_Ji;
                break;
            }
        }
        // CSR lower: row b -> col a (conjugate)
        for (int k = s->csr_row_ptr[b]; k < s->csr_row_ptr[b+1]; k++) {
            if (s->csr_col_idx[k] == a) {
                s->csr_Jr[k] = new_Jr;
                s->csr_Ji[k] = -new_Ji;
                break;
            }
        }
        // Dense J if allocated (eigenvalue computation)
        if (s->J_allocated) {
            J_RE(s, a, b) = new_Jr;
            J_IM(s, a, b) = new_Ji;
            J_RE(s, b, a) = new_Jr;
            J_IM(s, b, a) = -new_Ji;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Update Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double S2_of_spin(system_state *s, int a) {
    double sum = 0.0;
    for (int j = 0; j < s->D; j++) {
        double r = S_RE(s, a, j);
        double im = S_IM(s, a, j);
        sum += r*r + im*im;
    }
    return sum;
}

double Y2eq_of_spin(system_state *s, int a) {
    double r0 = Y_RE(s, a, 0), im0 = Y_IM(s, a, 0);
    double r1 = Y_RE(s, a, 1), im1 = Y_IM(s, a, 1);
    return (r0*r0 + im0*im0) - (r1*r1 + im1*im1);
}

void compute_hyperspin_norm(system_state *s) {
    for (int i = 0; i < s->N; i++) {
        S2(s, i) = S2_of_spin(s, i);
    }
}

// Precompute J*S coupling product: JS[a*D+j] = sum_b J[a*N+b] * S[b*D+j]
// Called once per RK2 half-step with either S or DS as source
// OpenMP: parallelized over the outer 'a' loop (each thread writes disjoint slice)
void compute_coupling_product_flat(const system_state *s,
    const double *src_re, const double *src_im,
    double *out_re, double *out_im)
{
    int N = s->N, D = s->D;
    const int *row_ptr = s->csr_row_ptr, *col_idx = s->csr_col_idx;
    const double *csr_Jr = s->csr_Jr, *csr_Ji = s->csr_Ji;

    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int a = 0; a < N; a++) {
        int a_off = a * D;
        for (int j = 0; j < D; j++) {
            out_re[a_off + j] = 0.0;
            out_im[a_off + j] = 0.0;
        }
        for (int k = row_ptr[a]; k < row_ptr[a+1]; k++) {
            int b = col_idx[k];
            double Jr_ab = csr_Jr[k];
            double Ji_ab = csr_Ji[k];
            int b_off = b * D;
            for (int j = 0; j < D; j++) {
                out_re[a_off + j] += Jr_ab * src_re[b_off + j] - Ji_ab * src_im[b_off + j];
                out_im[a_off + j] += Jr_ab * src_im[b_off + j] + Ji_ab * src_re[b_off + j];
            }
        }
    }
}

void RK2_step(system_state *s, double dt, double g, double beta, double deq, int coup_control, int eq_control) {

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MIDPOINT PHASE

    // Precompute J*S once for all spins and components
    compute_coupling_product_flat(s, s->S_re, s->S_im, s->JS_re, s->JS_im);

    // OpenMP: each spin update is independent once JS is precomputed
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int a = 0; a < s->N; a++) {
        int left  = (a + s->N - 1) % s->N;
        int next  = (a + 1) % s->N;

        // Use precomputed trig values
        double cphi = s->cos_phi[a];
        double sphi = s->sin_phi[a];

        // Non-interacting coefficients (computed once per spin, reused across components)
        double f0_coeff_re = 0.5 * ((0.5 * s->h[a] * cphi - g) - beta * S2(s, a));
        double f0_coeff_im = 0.5 * (-(0.5 * s->h[a] * cphi + g) - beta * S2(s, a));
        double f0_cross    = 0.5*0.5 * s->h[a] * sphi;

        // Equalization coefficient (computed once per spin)
        double feq_coeff = 0.0;
        if (eq_control) {
            feq_coeff = 0.5 * deq * (-Y2eq_of_spin(s, left) + Y2eq_of_spin(s, a));
        }

        for (int j = 0; j < s->D; j++) {
            double sr = S_RE(s, a, j);
            double si = S_IM(s, a, j);

            // Non-interacting term
            double f0R = f0_coeff_re * sr + f0_cross * si;
            double f0I = f0_coeff_im * si + f0_cross * sr;

            // Linear interaction term (precomputed, halve for midpoint)
            double fLR = 0, fLI = 0;
            if (coup_control) {
                fLR = 0.5 * JS_RE(s, a, j);
                fLI = 0.5 * JS_IM(s, a, j);
            }

            // Equalization term
            double feqR = feq_coeff * sr;
            double feqI = feq_coeff * si;

            DS_RE(s, a, j) = sr + 0.5 * dt * (f0R + fLR + feqR);
            DS_IM(s, a, j) = si + 0.5 * dt * (f0I + fLI + feqI);
        }

        // Equalizers derivatives (local vars per iteration, thread-safe)
        double dY0R = 0, dY0I = 0, dY1R = 0, dY1I = 0;
        if (eq_control) {
            double diffS2 = 0.5 * deq * (-S2(s, a) + S2(s, next));
            dY0R = diffS2 * Y_RE(s, a, 0);
            dY0I = diffS2 * Y_IM(s, a, 0);
            dY1R = -diffS2 * Y_RE(s, a, 1);
            dY1I = -diffS2 * Y_IM(s, a, 1);
        }
        DY_RE(s, a, 0) = Y_RE(s, a, 0) + 0.5 * dt * dY0R;
        DY_IM(s, a, 0) = Y_IM(s, a, 0) + 0.5 * dt * dY0I;
        DY_RE(s, a, 1) = Y_RE(s, a, 1) + 0.5 * dt * dY1R;
        DY_IM(s, a, 1) = Y_IM(s, a, 1) + 0.5 * dt * dY1I;
    }

    // Compute midpoint norms (sequential: small loop, not worth parallelizing)
    for (int i = 0; i < s->N; i++) {
        DS2(s, i) = 0.0;
        for (int j = 0; j < s->D; j++) {
            double r = DS_RE(s, i, j);
            double im = DS_IM(s, i, j);
            DS2(s, i) += r*r + im*im;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UPDATE PHASE

    // Precompute J*DS once for all spins and components
    compute_coupling_product_flat(s, s->DS_re, s->DS_im, s->JS_re, s->JS_im);

    // OpenMP: each spin update is independent once JS is precomputed
    #pragma omp parallel for schedule(static) if(use_parallel)
    for (int a = 0; a < s->N; a++) {
        int left  = (a + s->N - 1) % s->N;
        int next  = (a + 1) % s->N;

        // Use precomputed trig values (quick win)
        double cphi = s->cos_phi[a];
        double sphi = s->sin_phi[a];

        // Non-interacting coefficients (using DS norms)
        double f0_coeff_re = 0.5 * ((0.5 * s->h[a] * cphi - g) - beta * DS2(s, a));
        double f0_coeff_im = 0.5 * (-(0.5 * s->h[a] * cphi + g) - beta * DS2(s, a));
        double f0_cross    = 0.5*0.5 * s->h[a] * sphi;

        // Equalization coefficient (using DY)
        double feq_coeff = 0.0;
        if (eq_control) {
            double DY2_left = (DY_RE(s, left, 0)*DY_RE(s, left, 0) + DY_IM(s, left, 0)*DY_IM(s, left, 0))
                           - (DY_RE(s, left, 1)*DY_RE(s, left, 1) + DY_IM(s, left, 1)*DY_IM(s, left, 1));
            double DY2_right = (DY_RE(s, a, 0)*DY_RE(s, a, 0) + DY_IM(s, a, 0)*DY_IM(s, a, 0))
                            - (DY_RE(s, a, 1)*DY_RE(s, a, 1) + DY_IM(s, a, 1)*DY_IM(s, a, 1));
            feq_coeff = 0.5 * deq * (-DY2_left + DY2_right);
        }

        for (int j = 0; j < s->D; j++) {
            double sr = DS_RE(s, a, j);
            double si = DS_IM(s, a, j);

            double f0R = f0_coeff_re * sr + f0_cross * si;
            double f0I = f0_coeff_im * si + f0_cross * sr;

            double fLR = 0, fLI = 0;
            if (coup_control) {
                fLR = 0.5 * JS_RE(s, a, j);
                fLI = 0.5 * JS_IM(s, a, j);
            }

            double feqR = feq_coeff * sr;
            double feqI = feq_coeff * si;

            S_RE(s, a, j) += dt * (f0R + fLR + feqR);
            S_IM(s, a, j) += dt * (f0I + fLI + feqI);
        }

        // Equalizers update (local vars per iteration, thread-safe)
        double dY0R = 0, dY0I = 0, dY1R = 0, dY1I = 0;
        if (eq_control) {
            double diffS2 = 0.5 * deq * (-DS2(s, a) + DS2(s, next));
            dY0R = diffS2 * DY_RE(s, a, 0);
            dY0I = diffS2 * DY_IM(s, a, 0);
            dY1R = -diffS2 * DY_RE(s, a, 1);
            dY1I = -diffS2 * DY_IM(s, a, 1);
        }
        Y_RE(s, a, 0) += dt * dY0R;
        Y_IM(s, a, 0) += dt * dY0I;
        Y_RE(s, a, 1) += dt * dY1R;
        Y_IM(s, a, 1) += dt * dY1I;
    }

    // Compute final norms
    compute_hyperspin_norm(s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Physical Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double max_eigenvalue_symm(double *data, int n) {
    gsl_matrix_view m = gsl_matrix_view_array(data, n, n);
    gsl_vector *eval = gsl_vector_alloc(n);
    gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(n);
    gsl_eigen_symm(&m.matrix, eval, w);
    gsl_eigen_symm_free(w);

    double max = gsl_vector_get(eval, 0);
    for (int i = 1; i < n; i++) {
        double v = gsl_vector_get(eval, i);
        if (v > max) max = v;
    }
    gsl_vector_free(eval);
    return max;
}

double Linearized_Max_Eigenvalue(int N, double *J_re, double *J_im, double *h, double g) {
    int dim = 2 * N;
    double *Lmat = calloc(dim * dim, sizeof(double));
    if (!Lmat) { fprintf(stderr, "Memory allocation failed in Linearized_Max_Eigenvalue\n"); return 0; }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j <= i; j++) {
            double val;
            if (i < N) {
                if (j == i) {
                    val = 0.5*(0.5*h[i] - g);
                } else {
                    val = 0.5*J_re[i*N + j];
                }
            } else {
                if (j == i) {
                    val = 0.5*(-0.5*h[i-N] - g);
                } else {
                    if (j < N) {
                        val = -0.5*J_im[(i-N)*N + j];
                    } else {
                        val = +0.5*J_re[(i-N)*N + (j-N)];
                    }
                }
            }
            Lmat[i*dim + j] = val;
            if (j != i) Lmat[j*dim + i] = val;
        }
    }
    double result = max_eigenvalue_symm(Lmat, dim);
    free(Lmat);
    return result;
}

double h_threshold(int N, double g, double *J_re, double *J_im) {
    int dim = 2 * N;
    double *Lmat = calloc(dim * dim, sizeof(double));
    if (!Lmat) { fprintf(stderr, "Memory allocation failed in h_threshold\n"); return 0; }

    double h_val = 0, dh_val = 0.00001, hmax = 5, lmax = 0;

    while (lmax <= 0 && h_val < hmax) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j <= i; j++) {
                double val;
                if (i < N) {
                    val = (j == i) ? 0.5*(0.5*h_val - g) : 0.5*J_re[i*N + j];
                } else {
                    if (j == i) {
                        val = 0.5*(-0.5*h_val - g);
                    } else {
                        val = (j < N) ? -0.5*J_im[(i-N)*N + j] : +0.5*J_re[(i-N)*N + (j-N)];
                    }
                }
                Lmat[i*dim + j] = val;
                if (j != i) Lmat[j*dim + i] = val;
            }
        }
        lmax = max_eigenvalue_symm(Lmat, dim);
        h_val += dh_val;
    }
    free(Lmat);
    double hth = h_val - dh_val;
    printf("> hth  = %0.4f\n", hth);
    printf("> lmax = %0.4f  (max linearized matrix eigenvalue)\n", lmax);
    return hth;
}

observables compute_observables(system_state *s, double g, double beta, double dt, double *L_old, int compute_H, int compute_full) {
    observables o = {0};
    int N = s->N, D = s->D, num_edges = s->num_edges;
    double *h = s->h;
    double *inv_sqrtS2 = s->inv_sqrtS2;

    // Shared scratch: computed only when H or m is requested
    if (compute_H) {
        #pragma omp parallel for schedule(static) if(use_parallel)
        for (int i = 0; i < N; i++)
            inv_sqrtS2[i] = 1.0 / sqrt(S2(s, i));
    }

    if (compute_full) {
        // Merged spin pass: Ly0 + Average_Amplitude_Variance
        double L0_acc = 0, mean_S2 = 0, mean2_S2 = 0;
        #pragma omp parallel for schedule(static) if(use_parallel) reduction(+:L0_acc,mean_S2,mean2_S2)
        for (int a = 0; a < N; a++) {
            double s2 = S2(s, a);
            double Lksqsum = 0;
            for (int k = 0; k < D; k++) {
                double r = S_RE(s, a, k), im = S_IM(s, a, k);
                Lksqsum += r*r - im*im;
            }
            L0_acc -= 0.5*(0.5*h[a]*Lksqsum - g*s2 - 0.5*beta*s2*s2);
            mean_S2 += s2;
            mean2_S2 += s2*s2;
        }
        o.L0 = L0_acc;
        mean_S2 /= N;
        mean2_S2 /= N;
        o.DS2 = fabs(mean2_S2 - mean_S2*mean_S2) / N;
    }

    // Merged edge pass: H (computed only when requested) + Lyc (conditional)
    if (compute_H) {
        double H_acc = 0, Lc_acc = 0;
        #pragma omp parallel for schedule(static) if(use_parallel) reduction(+:H_acc,Lc_acc)
        for (int e = 0; e < num_edges; e++) {
            int p = s->edge_a[e], q = s->edge_b[e];
            double Jr = s->edge_Jr[e], Ji = s->edge_Ji[e];
            double ip_s2 = inv_sqrtS2[p], iq_s2 = inv_sqrtS2[q];
            for (int k = 0; k < D; k++) {
                double srp = S_RE(s, p, k), sip = S_IM(s, p, k);
                double srq = S_RE(s, q, k), siq = S_IM(s, q, k);
                double sRp = srp * ip_s2, sIp = sip * ip_s2;
                double sRq = srq * iq_s2, sIq = siq * iq_s2;
                H_acc -= 2.0 * (Jr*(sRp*sRq + sIp*sIq) - Ji*(sRp*sIq - sIp*sRq));
                if (compute_full)
                    Lc_acc -= Jr*(srp*srq + sip*siq) + Ji*(sip*srq - srp*siq);
            }
        }
        o.H = H_acc;
        o.Lc = Lc_acc;
    }

    if (compute_full) {
        o.L = o.L0 + o.Lc;
        o.dL = (o.L - *L_old) / dt;
        *L_old = o.L;

        // Magnetization (parallelized over spins N, not components D)
        double compR[D], compI[D];
        for (int k = 0; k < D; k++) { compR[k] = 0.0; compI[k] = 0.0; }
        #pragma omp parallel for schedule(static) if(use_parallel) reduction(+:compR[:D],compI[:D])
        for (int a = 0; a < N; a++) {
            double inv = inv_sqrtS2[a];
            for (int k = 0; k < D; k++) {
                compR[k] += S_RE(s, a, k) * inv;
                compI[k] += S_IM(s, a, k) * inv;
            }
        }
        double m_acc = 0.0;
        for (int k = 0; k < D; k++) {
            m_acc += compR[k]*compR[k] + compI[k]*compI[k];
        }
        o.m = sqrt(m_acc) / N;
    }

    return o;
}

int system_is_real(system_state *s) {
    int count_im=0;
     for (int a = 0; a < s->N; a++) {
        for (int j = 0; j < s->D; j++) {
            if ( fabs(S_IM(s, a, j)) > 1e-9 ) {count_im++;}
        }
     }
     if(count_im==0){return 0;}
     else{return 1;}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Output Management (Buffered I/O)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

io_context init_output_files(int N, int D, int save_dyn, int buffer_size, const char *base_dir) {
    io_context ctx;
    memset(&ctx, 0, sizeof(io_context));

    char path[1024];

    if (save_dyn == 0) {
        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Final.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Final.dat");
        ctx.f_final = fopen(path, "a");
        if (!ctx.f_final) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_final, "NS  T  ");
        for (int i = 0; i < N; i++) {
            fprintf(ctx.f_final, "S%d    ", i+1);
            for (int j = 0; j < D; j++) {
                fprintf(ctx.f_final, "Re(A%d%d)  Im(A%d%d)  ", i+1, j+1, i+1, j+1);
            }
        }
        fprintf(ctx.f_final, "\n");
        fflush(ctx.f_final);
        char *buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_final, buf, _IOFBF, buffer_size);
        return ctx;
    }

    if (save_dyn == 1) {
        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Fields.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Fields.dat");
        ctx.f_fields = fopen(path, "w");
        if (!ctx.f_fields) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_fields, "t        spin_id  S2       ");
        for (int j = 0; j < D; j++) fprintf(ctx.f_fields, "Re(A%d)   ", j+1);
        for (int j = 0; j < D; j++) fprintf(ctx.f_fields, "Im(A%d)   ", j+1);
        fprintf(ctx.f_fields, "\n");
        char *buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_fields, buf, _IOFBF, buffer_size);

        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Equalizers.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Equalizers.dat");
        ctx.f_equalizers = fopen(path, "w");
        if (!ctx.f_equalizers) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_equalizers, "t        ");
        for (int j = 0; j < N; j+=2) fprintf(ctx.f_equalizers, "Re(Y%d)    Re(Y%d)    ", j, j+1);
        for (int j = 0; j < N; j+=2) fprintf(ctx.f_equalizers, "Im(Y%d)    Im(Y%d)    ", j, j+1);
        fprintf(ctx.f_equalizers, "\n");
        buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_equalizers, buf, _IOFBF, buffer_size);

        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Hamilton.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Hamilton.dat");
        ctx.f_hamilton = fopen(path, "w");
        if (!ctx.f_hamilton) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_hamilton, "t    H\n");
        buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_hamilton, buf, _IOFBF, buffer_size);

        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Lyapunov.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Lyapunov.dat");
        ctx.f_lyapunov = fopen(path, "w");
        if (!ctx.f_lyapunov) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_lyapunov, "t    L    dL    L0    Lc\n");
        buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_lyapunov, buf, _IOFBF, buffer_size);
    }

    if (save_dyn == 2) {
        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Hamilton.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Hamilton.dat");
        ctx.f_hamilton = fopen(path, "w");
        if (!ctx.f_hamilton) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_hamilton, "t    H\n");
        char *buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_hamilton, buf, _IOFBF, buffer_size);

        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_Lyapunov.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_Lyapunov.dat");
        ctx.f_lyapunov = fopen(path, "w");
        if (!ctx.f_lyapunov) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_lyapunov, "t    L    dL    L0    Lc\n");
        buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_lyapunov, buf, _IOFBF, buffer_size);
    }

    if (save_dyn == 3) {
        if (base_dir && base_dir[0])
            snprintf(path, sizeof(path), "%s/Data_H_Ly_m_DS2.dat", base_dir);
        else
            snprintf(path, sizeof(path), "Data_H_Ly_m_DS2.dat");
        ctx.f_h_ly_m_ds2 = fopen(path, "w");
        if (!ctx.f_h_ly_m_ds2) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
        fprintf(ctx.f_h_ly_m_ds2, "t    H    Ly/N    <m>    <DS2>\n");
        char *buf = malloc(buffer_size);
        if (buf) setvbuf(ctx.f_h_ly_m_ds2, buf, _IOFBF, buffer_size);
    }

    // Final config file
    if (base_dir && base_dir[0])
        snprintf(path, sizeof(path), "%s/Data_Final.dat", base_dir);
    else
        snprintf(path, sizeof(path), "Data_Final.dat");
    ctx.f_final = fopen(path, "w");
    if (!ctx.f_final) { fprintf(stderr, "Error creating %s\n", path); return ctx; }
    fprintf(ctx.f_final, "NS  T  ");
    for (int i = 0; i < N; i++) {
        fprintf(ctx.f_final, "S%d    ", i+1);
        for (int j = 0; j < D; j++) {
            fprintf(ctx.f_final, "Re(A%d%d)  Im(A%d%d)  ", i+1, j+1, i+1, j+1);
        }
    }
    fprintf(ctx.f_final, "\n");
    char *buf = malloc(buffer_size);
    if (buf) setvbuf(ctx.f_final, buf, _IOFBF, buffer_size);

    return ctx;
}

void save_output_files(io_context *ctx, int save_dyn, system_state *s, double t, double L, double L0, double Lc, double dL, double H, double m, double DS2) {
    int N = s->N, D = s->D;

    if (save_dyn == 1) {
        for (int i = 0; i < N; i++) {
            fprintf(ctx->f_fields, "%0.15f %d %0.15f ", t, i+1, S2(s, i));
            for (int j = 0; j < D; j++) fprintf(ctx->f_fields, "%0.15f ", S_RE(s, i, j));
            for (int j = 0; j < D; j++) fprintf(ctx->f_fields, "%0.15f ", S_IM(s, i, j));
            fprintf(ctx->f_fields, "\n");
        }

        fprintf(ctx->f_equalizers, "%0.15f ", t);
        for (int j = 0; j < N; j+=2)
            fprintf(ctx->f_equalizers, "%0.15f %0.15f ", Y_RE(s, j, 0), Y_RE(s, j, 1));
        for (int j = 0; j < N; j+=2)
            fprintf(ctx->f_equalizers, "%0.15f %0.15f ", Y_IM(s, j, 0), Y_IM(s, j, 1));
        fprintf(ctx->f_equalizers, "\n");
    }

    if (save_dyn == 1 || save_dyn == 2) {
        fprintf(ctx->f_hamilton, "%0.15f  %0.15f\n", t, H);
        fprintf(ctx->f_lyapunov, "%0.15f %0.15f %0.15f %0.15f %0.15f\n", t, L, dL, L0, Lc);
    }

    if (save_dyn == 3) {
        fprintf(ctx->f_h_ly_m_ds2, "%0.15f  %0.15f  %0.15f  %0.15f  %0.15f\n", t, H, L/N, m, DS2);
    }
}

void flush_output_files(io_context *ctx, int save_dyn) {
    if (save_dyn == 1) {
        fflush(ctx->f_fields); fflush(ctx->f_equalizers);
        fflush(ctx->f_hamilton); fflush(ctx->f_lyapunov);
    }
    if (save_dyn == 2) { fflush(ctx->f_hamilton); fflush(ctx->f_lyapunov); }
    if (save_dyn == 3) { fflush(ctx->f_h_ly_m_ds2); }
    fflush(ctx->f_final);
}

void close_output_files(io_context *ctx, int save_dyn) {
    if (save_dyn == 1) {
        if (ctx->f_fields) fclose(ctx->f_fields);
        if (ctx->f_equalizers) fclose(ctx->f_equalizers);
        if (ctx->f_hamilton) fclose(ctx->f_hamilton);
        if (ctx->f_lyapunov) fclose(ctx->f_lyapunov);
    }
    if (save_dyn == 2) {
        if (ctx->f_hamilton) fclose(ctx->f_hamilton);
        if (ctx->f_lyapunov) fclose(ctx->f_lyapunov);
    }
    if (save_dyn == 3) {
        if (ctx->f_h_ly_m_ds2) fclose(ctx->f_h_ly_m_ds2);
    }
    if (ctx->f_final) fclose(ctx->f_final);
}

void save_final_config(io_context *ctx, system_state *s, double t, int NS) {
    if (!ctx->f_final) { fprintf(stderr, "Error: final output file not open\n"); return; }
    int N = s->N, D = s->D;
    fprintf(ctx->f_final, "%d  %0.15f  ", NS, t);
    for (int i = 0; i < N; i++) {
        fprintf(ctx->f_final, "%0.15f  ", S2(s, i));
        for (int j = 0; j < D; j++) {
            fprintf(ctx->f_final, "%0.15f  %0.15f  ", S_RE(s, i, j), S_IM(s, i, j));
        }
    }
    fprintf(ctx->f_final, "\n");
    fflush(ctx->f_final);
}

//Read entire file into a malloc'd null-terminated string. Caller must free.
char *read_file_to_string(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    if (len <= 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

//FNV-1a 64-bit hash of ini content + seed + matrix content
unsigned long long compute_hash(const char *ini_path, int seed, const char *matrix_path) {
    unsigned long long h = 14695981039346656037ULL;
    char *ini = read_file_to_string(ini_path);
    if (ini) {
        for (const char *p = ini; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
        free(ini);
    }
    char seed_str[32];
    snprintf(seed_str, sizeof(seed_str), "%d", seed);
    for (const char *p = seed_str; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    char *mat = read_file_to_string(matrix_path);
    if (mat) {
        for (const char *p = mat; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
        free(mat);
    }
    return h;
}

void format_hash_hex(unsigned long long hash, char hex[17]) {
    for (int i = 15; i >= 0; i--) {
        unsigned char nibble = (unsigned char)(hash & 0xF);
        hex[i] = nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
        hash >>= 4;
    }
    hex[16] = '\0';
}

int copy_file_to_dir(const char *src, const char *dir, const char *dst_name) {
    FILE *in = fopen(src, "rb");
    if (!in) return 1;
    char dst[1024];
    if (dir && dir[0])
        snprintf(dst, sizeof(dst), "%s/%s", dir, dst_name);
    else
        snprintf(dst, sizeof(dst), "%s", dst_name);
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return 1; }
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) {
        if (fwrite(buf, 1, n, out) != n) { fclose(in); fclose(out); return 1; }
    }
    fclose(in);
    fclose(out);
    return 0;
}

void write_metadata(const char *dir, int NS, int N, int D, int initial_type, double pth, double hm, double H, double t_final, const char *stop_reason) {
    if (!dir || !dir[0]) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/metadata.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) return;
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    fprintf(f, "Complex_HSM v2.9\n");
    fprintf(f, "Run time: %s\n", time_str);
    fprintf(f, "NS=%d N=%d D=%d initial_type=%d pth=%.3f hm=%.6f\n", NS, N, D, initial_type, pth, hm);
    fprintf(f, "H_final=%.15f t_final=%.6f\n", H, t_final);
    fprintf(f, "Stop reason: %s\n", stop_reason);
    fprintf(f, "\nOutput files:\n");
    fprintf(f, "  Data_Final.dat\n");
    fprintf(f, "  Data_Matrix.dat (coupling matrix)\n");
    fprintf(f, "  Complex_HSM.ini (configuration used)\n");
    fclose(f);
}
