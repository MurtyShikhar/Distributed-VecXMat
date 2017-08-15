#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include <time.h>

#define ROOT 0
#define DUMMY_TAG 0
#define EPSILON 0.000001
#define SPEEDUP 8 

typedef long ll;
typedef long long lll;

using namespace std;
const int num_streams = 4;

double PARTITION_RATIO;
int gpu_procs;
int inspect_id;
int use_streams;
int optimize_gpu_usage;

struct Entry{
	int row;
	int col;
	ll val;
	Entry(int r,int c,ll v){
		row = r;
		col = c;
		val = v;
	}
};

struct Input{
	vector<Entry> nz;
	vector<ll> vec;
	string name;
};

struct Result{
	int dim; // number of columns
	int n_rows; // number of rows
	int nnz;  // number of non zero elements
	ll* vec; // vector
	bool isVecNNZ; // sparse or complete 
	int* ptr; // matrix
	int* indices; // matrix
	ll* data; // matrix
	string output_name;
};


/*
Input assumed to be corresponding to square matrix
*/
Input parse_file(char* filename){

	ifstream input_file(filename);
	string line,token,name;
	int dim;
	string V = "B";
	
	vector<Entry> A;
	vector<ll> B;
	Input res;
	if(input_file.is_open()){
		
		//read file name
		getline(input_file,line);
		stringstream ss1(line);
		ss1 >> token;
		ss1 >> name;

		//get dimension
		getline(input_file,line);
		stringstream ss2(line);
		ss2 >> token;
		ss2 >> dim;

		
		getline(input_file,line);
		getline(input_file,line);

		//read non-zero elements of matrix A
		int r,c;
		ll v;
		while(line.compare(V)){
			stringstream ss(line);
			ss >> r;
			ss >> c;
			ss >> v;
			Entry tmp(r,c,v);
			A.push_back(tmp);
			getline(input_file,line);
		}

		//read vector B
		B.resize(dim);
		for(int i = 0; i < dim; ++i)
		{
			getline(input_file,line);
			v = atoll(line.c_str());
			B[i] = v;
		}
		input_file.close();

		res.nz = A;
		res.vec = B;
		res.name = name;
	}
	else{
		cout << "Couldn't open file" <<endl;
	}
	return res;
}

void get_csr_representation(int n_rows,vector<Entry>* nz,int* ptr,int* indices,ll* data){
	int prev_row_index = 0;
	int curr_row_index = 0;
	ptr[0] = 0;
	int i;
	
	for(i = 0; i < nz->size(); ++i)
	{
		curr_row_index = nz->at(i).row;
		if(curr_row_index != prev_row_index){
			for(int j = prev_row_index+1; j <= curr_row_index; ++j)
				ptr[j] = i;

			prev_row_index = curr_row_index;
		}
		indices[i] = nz->at(i).col;
		data[i] = nz->at(i).val;	
	}
	for(int i = curr_row_index+1; i <= n_rows; ++i)
	{
		ptr[i] = nz->size();		
	}
}

void print_res(int proc_id, Result res){
	cout << "curr_process " << proc_id << endl; 
	cout << "Dimension: " << res.dim << endl;
	cout << "NNZ: " << res.nnz << endl;
	cout << "Rows: " << res.n_rows << endl;


	//print vector 
	cout << "Printing vector:- " << endl;
	int sz = res.dim;
	if(res.isVecNNZ)
		sz = res.nnz;
	for(int i = 0; i < sz; ++i)
	{
		cout << res.vec[i] << ",";
	}
	cout << endl;

	cout << "Printing ptr" << endl;
	for(int i = 0; i <= res.n_rows; ++i)
	{
		cout << res.ptr[i] << ",";
	}
	cout << endl;

	cout << "Printing indices" << endl;
	for(int i = 0; i < res.nnz; ++i)
	{
		cout << res.indices[i] << ",";
	}
	cout << endl;

	cout << "Printing data" << endl;
	for(int i = 0; i < res.nnz; ++i)
	{
		cout << res.data[i] << ",";
	}
	cout << endl;


	cout << "\n";
	cout <<"Name: " << res.output_name << endl;
	cout << "-----------------------------------------------\n";
}

void printVec(ll* out, int sz) {
	cout << "Result:\n" ;
	for (int i=0; i < sz; i++) cout << out[i] << ",";
	cout << "\n";
}

void write_data(lll* product, int n, const char* fname) {
	ofstream fout(fname);
	for (int i= 0; i < n; i++) 
		fout << product[i] << "\n";

	fout.close();
} 
Result read_data(char* filename){
	Input parse = parse_file(filename);

	Result res;
	res.output_name = "Output_"+parse.name;
	res.dim = parse.vec.size();
	res.nnz = parse.nz.size();

	//convert data representation to CSR format
	res.ptr = (int*) malloc((res.dim+1)*sizeof(int));
	res.indices = (int*) malloc(res.nnz*sizeof(int));
	res.data = (ll*) malloc(res.nnz*sizeof(ll));
	
	ll* vec = (ll*) malloc((res.dim)*sizeof(ll));
	for(int i = 0; i < res.dim; ++i)
	{
		vec[i] = parse.vec[i];
	}
	res.vec = vec;
	res.n_rows = res.dim;
	res.isVecNNZ = false;
	get_csr_representation(res.dim,&parse.nz,res.ptr,res.indices,res.data);

	return res;
}


__global__ void dot_prod_kernel_warp(const int num_rows, const int* ptr, const int* indices, const ll* data, const ll* vec, lll * out) {
    int tid = threadIdx.x + (blockDim.x*blockIdx.x);
	int warp_id = tid/32;
	int tid_warp = tid % 32;

	__shared__  lll vals[256];
	if (warp_id < num_rows) {
		int row_start = ptr[warp_id]-ptr[0];
		int row_end = ptr[warp_id+1]-ptr[0];
		lll local_sum = 0;
		
		for (int i = row_start + tid_warp; i < row_end; i += 32) 
			local_sum += data[i]*vec[i];

		vals[threadIdx.x] = local_sum;
		/* massive control flow divergence within warp... */
		if ( tid_warp < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
		if ( tid_warp < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
		if ( tid_warp < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
		if ( tid_warp < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
		if ( tid_warp < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];

		if ( tid_warp == 0)
			out[warp_id] = vals[threadIdx.x];
	}

}


__global__ void dot_prod_kernel ( const int num_rows , const int * ptr , const int * indices ,const ll * data , const ll * vec ,ll * out)
{
    int tid = threadIdx.x + (blockDim.x*blockIdx.x);


    if (tid < num_rows) {
        int start = ptr[tid] - ptr[0];
        int en = ptr[tid+1] - ptr[0]; 
		ll sum = 0.0;
		//printf("%d handling %d\n", tid, en-start);
        for (int i = start; i < en; i++) {
            sum += vec[i]*data[i];
        }

        out[tid] = sum;
    }

}


void compute_on_gpu_helper(Result& res, cudaStream_t& stream, lll* out_local,  int i, int *d_ptr, int *d_indices, ll *d_data,ll *d_vec, lll *d_out) {
	 
	 int st_row = i*res.n_rows/num_streams;
	 int end_row = (i == num_streams-1)?  res.n_rows : (i+1)*res.n_rows/num_streams;
	/* current stream dealing with everything from [st_row, en_row) */
	 int num_rows = end_row - st_row;
	 int st_idx = res.ptr[st_row] - res.ptr[0];
	 int en_idx = res.ptr[end_row] -res.ptr[0];

	 int num_elems = en_idx - st_idx;

	if (num_elems == 0) {
		for (int i = st_row; i <end_row; i++) out_local[i] = 0;
		return;
	}



    cudaMemcpyAsync(d_ptr, &res.ptr[st_row], sizeof(int)*(num_rows+1), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_indices, &res.indices[st_idx], sizeof(int)*num_elems, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_data, &res.data[st_idx], sizeof(ll)*num_elems, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vec, &res.vec[st_idx], sizeof(ll)*num_elems, cudaMemcpyHostToDevice, stream);

    dot_prod_kernel_warp<<< 1 + (num_rows/8), 256, 0, stream>>>(num_rows, d_ptr, d_indices, d_data, d_vec, d_out);


    cudaMemcpyAsync(&out_local[st_row], d_out, sizeof(lll)*num_rows, cudaMemcpyDeviceToHost, stream);


}

void compute_on_gpu_helper_no_streams(Result& res, lll* out_local,int proc_id){
	/* current stream dealing with everything from [st_row, en_row) */
	int num_rows = res.n_rows;
	int num_elems = res.nnz;
	int *d_ptr, *d_indices;
	ll *d_data,  *d_vec;
	lll *d_out;

    cudaMalloc((void**)&d_ptr, sizeof(int)*(num_rows+1));
    cudaMalloc((void**)&d_indices, sizeof(int)*num_elems);
    cudaMalloc((void**)&d_data, sizeof(ll)*num_elems);
    cudaMalloc((void**)&d_out, sizeof(lll)*num_rows);
    cudaMalloc((void**)&d_vec, sizeof(ll)*num_elems);


    cudaMemcpy(d_ptr, res.ptr, sizeof(int)*(num_rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, res.indices, sizeof(int)*num_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, res.data, sizeof(ll)*num_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, res.vec, sizeof(ll)*num_elems, cudaMemcpyHostToDevice);

    dot_prod_kernel_warp<<< 1 + (num_rows/8), 256>>>(num_rows, d_ptr, d_indices, d_data, d_vec, d_out);


    cudaMemcpy(out_local, d_out, sizeof(lll)*num_rows, cudaMemcpyDeviceToHost);

}

void compute_on_gpu(Result& res, lll* out_local, int proc_id ) {
	
	int num_elems = res.nnz;
	int num_rows = res.n_rows;
	if (num_elems == 0) {
		for (int i = 0; i < res.n_rows; i++) out_local[i] = 0;
		return;
	}

	int num_devices;
	cudaGetDeviceCount(&num_devices);
	printf("node has %d devices\n", num_devices);
	cudaSetDevice(proc_id%num_devices);
	if (!use_streams){
		compute_on_gpu_helper_no_streams(res, out_local, proc_id);
		return;
	}

	int *d_ptr, *d_indices;
	ll *d_data, *d_out, *d_vec;
    cudaMalloc((void**)&d_ptr, sizeof(int)*(num_rows+1));
    cudaMalloc((void**)&d_indices, sizeof(int)*num_elems);
    cudaMalloc((void**)&d_data, sizeof(ll)*num_elems);
    cudaMalloc((void**)&d_out, sizeof(lll)*num_rows);
    cudaMalloc((void**)&d_vec, sizeof(ll)*num_elems);
	cudaStream_t streams[num_streams];
	
	for (int i=0; i < num_streams; i++) {
	 	int st_row = i*res.n_rows/num_streams;
	 	int st_idx = res.ptr[st_row] - res.ptr[0];
		cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
		
		compute_on_gpu_helper(res, streams[i], out_local, i, (int*)&d_ptr[st_row], (int*)&d_indices[st_idx],(ll*)&d_data[st_idx], (ll*)&d_vec[st_idx], (lll*)&d_out[st_row]);
	}

	cudaDeviceSynchronize();
	return;

}


void compute_on_cpu(Result& res,lll*  out_local, int proc_id){
	int num_rows = res.n_rows;

	for (int i=0; i < num_rows; i++) {
		lll sum = 0;
		int start = res.ptr[i] - res.ptr[0];
		int en = res.ptr[i+1] - res.ptr[0];
		for (int j = start; j < en; j++) 
			sum += res.data[j]*res.vec[j];

		out_local[i] = sum;
	}


}


//returns indices to send to other node
pair<int,int> get_partition_index(int nrows,int GPU_present){
//review logic
	int i,j,r;
	double d = PARTITION_RATIO*nrows;
	r = (int)d;
	if (abs(d-(double)r) > EPSILON){
		r++;
	}

	if(GPU_present){
		i = r;
		j = nrows-1;
	}
	else{
		i = 0;
		j = r-1;
	}
	return make_pair(i,j);
}

int get_proc_load(const int id,const int nproc,const int n_rows){
	int i = n_rows/nproc;
	int r = n_rows%nproc;
	if(id < r)	
		return i+1;
	else
		return i;
}

		
void sendDataToRoot(Result& res, int root_id,int GPU_present){
	//decide partition to send to second root
	pair<int,int> partition_idx = get_partition_index(res.dim,GPU_present);
	int start_row = partition_idx.first;
	int end_row = partition_idx.second;
	int sz = end_row-start_row+1;
	int s = res.ptr[start_row], e = res.ptr[end_row+1];
	int count = e-s;
	// send dim
	MPI_Send(&res.dim,1,MPI_INT,root_id,DUMMY_TAG,MPI_COMM_WORLD);
	//send ptr
	MPI_Send(&res.ptr[partition_idx.first],sz,MPI_INT,root_id,DUMMY_TAG,MPI_COMM_WORLD);

	//send count
	MPI_Send(&count,1,MPI_INT,root_id,DUMMY_TAG,MPI_COMM_WORLD);
	// send indices
	MPI_Send(&res.indices[s],e-s,MPI_INT,root_id,DUMMY_TAG,MPI_COMM_WORLD);
	// send data
	MPI_Send(&res.data[s],e-s,MPI_LONG,root_id,DUMMY_TAG,MPI_COMM_WORLD);
	// send vector
	MPI_Send(res.vec,res.dim,MPI_LONG,root_id,DUMMY_TAG,MPI_COMM_WORLD);

	res.nnz -= count;
	res.n_rows -= sz;

	if (partition_idx.first == 0) {
		res.indices = &res.indices[e];
		res.data = &res.data[e];
		res.ptr = &res.ptr[end_row+1];
	}
}

void recvDataFromRoot(Result& res,int GPU_present){
	int count;	//number of nnz elements received	
	MPI_Recv(&res.dim,1,MPI_INT,ROOT,DUMMY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	pair<int,int> partition_idx = get_partition_index(res.dim,1-GPU_present);
	int recv_size = partition_idx.second - partition_idx.first+1;
	res.ptr = (int*) malloc(sizeof(int)*recv_size+1);

	MPI_Recv(res.ptr,recv_size,MPI_INT,ROOT,DUMMY_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(&count,1,MPI_INT,ROOT,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	res.ptr[recv_size] = count+res.ptr[0];

	res.indices = (int*)malloc(sizeof(int)*count);
	res.data = (ll*)malloc(sizeof(ll)*count);
	res.vec = (ll*)malloc(sizeof(ll)*res.dim);
	res.nnz = count;
	res.n_rows =recv_size;

	MPI_Recv(res.indices,count,MPI_INT,ROOT,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(res.data,count,MPI_LONG,ROOT,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(res.vec,res.dim,MPI_LONG,ROOT,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
}

/* m x n matrix  */
void sendDataToGroup(Result& res, int GPU_present, lll* out, int orig_id,  MPI_Comm load){
	
	int* counts;
    int* counts_rows;
    int* displ;
    int* ptr_local;
    ll* vec_nnz;
    ll* vec_local;
    int* indices_local;
    ll* data_local;
    lll* out_local;
    int* per_proc_load;

	int proc_id,nproc;
	MPI_Comm_rank(load,&proc_id);
	MPI_Comm_size(load,&nproc);
	int total_rows;

	if (proc_id == 0) total_rows = res.n_rows;

	MPI_Bcast(&res.dim,1,MPI_INT,0,load); // broadcast the column size to every process
	MPI_Bcast(&total_rows,1,MPI_INT,0,load); // broadcast the column size to every process


    res.n_rows = get_proc_load(proc_id,nproc,total_rows);   /*number of rows the processor is to handle*/ 
    
    if(proc_id == ROOT){
        counts_rows = (int*) malloc(sizeof(int)*nproc); // count_rows[i] stores the number of processes given to all processes from 0 to i (excluding i)
		per_proc_load = (int*) malloc(sizeof(int)*nproc); // stores the load assigned to each process
        counts = (int*) malloc(sizeof(int)*nproc);
        displ = (int*) malloc(sizeof(int)*nproc);
        vec_nnz = (ll*) malloc(sizeof(ll)*res.nnz);

        //populate counts_rows array
        int div = total_rows/nproc;
        int rem = total_rows%nproc;
        counts_rows[0] = 0;
		per_proc_load[0] = (rem > 0) ? div + 1 : div; 
        for (int i = 1; i < nproc; ++i)
        {
			counts_rows[i] = counts_rows[i-1] + per_proc_load[i-1];
			per_proc_load[i] = (i < rem) ? div + 1 : div; 
        }

        //scope for using openmp
        for (int p = 0; p < nproc; p++) {
            int index1,index2;
            if(p!= nproc-1)
                index1 = counts_rows[p+1];                 
            else
                index1 = total_rows;

            index2 = counts_rows[p];
            counts[p] = res.ptr[index1] - res.ptr[index2];
            displ[p] = res.ptr[index2] - res.ptr[0];
        } 

        for(int i = 0; i < res.nnz; ++i) vec_nnz[i] = res.vec[res.indices[i]];
    
    }

    ptr_local = (int *) malloc(sizeof(int)*(res.n_rows+1)); /* size of ptr = num_rows + 1 */
  
    MPI_Scatterv(res.ptr, per_proc_load, counts_rows, MPI_INT, ptr_local, res.n_rows+1, MPI_INT, 0, load);
    
    MPI_Scatter(counts, 1, MPI_INT, &res.nnz, 1, MPI_INT, 0, load);

    ptr_local[res.n_rows] = res.nnz + ptr_local[0];

    if (GPU_present){
		cudaMallocHost(&indices_local, sizeof(int)*res.nnz);
		cudaMallocHost(&vec_local, sizeof(ll)*res.nnz);
		cudaMallocHost(&data_local, sizeof(ll)*res.nnz);
		cudaMallocHost(&out_local, sizeof(lll)*res.n_rows);
	}

	else {
		indices_local = (int*) malloc(sizeof(int)*res.nnz);
		vec_local = (ll*) malloc(sizeof(ll)*res.nnz);
		data_local = (ll*) malloc(sizeof(ll)*res.nnz);
		out_local = (lll*) malloc(sizeof(lll)*res.n_rows);
	}



    MPI_Scatterv(res.indices,counts,displ,MPI_INT,indices_local,res.nnz,MPI_INT,0,load); 
    MPI_Scatterv(res.data,counts,displ,MPI_LONG,data_local,res.nnz,MPI_LONG,0,load); 
    MPI_Scatterv(vec_nnz,counts,displ,MPI_LONG,vec_local,res.nnz,MPI_LONG,0,load);
	


    res.data = data_local;
    res.indices = indices_local;
    res.ptr = ptr_local;
    res.vec = vec_local;
    res.isVecNNZ = true;



	if (GPU_present)
		compute_on_gpu(res,out_local, proc_id);
	else 
		compute_on_cpu(res, out_local, proc_id);


	if (orig_id == inspect_id) print_res(orig_id, res); 
	MPI_Gatherv(out_local, res.n_rows, MPI_LONG_LONG, out, per_proc_load, counts_rows, MPI_LONG_LONG, 0, load);	

   
}

void partition_and_compute(int GPU_present,int proc_id, int ngpus, Result& res, lll* out) {
	MPI_Comm load;
	MPI_Comm_split(MPI_COMM_WORLD,GPU_present,proc_id, &load);
	
	int nproc; MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	int ncpus = nproc - ngpus;
	double n = ncpus/ngpus;
	PARTITION_RATIO = SPEEDUP/(SPEEDUP + n);

	lll* out_local;

	// find the partition in which old root resides 
	int root_partition;
	if (proc_id == ROOT) {
		root_partition = GPU_present;
	}

	MPI_Bcast(&root_partition, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int rank_new;
	int second_root;
	bool is_second_root = false;
	
	/* other set */
	if (GPU_present != root_partition) {
		MPI_Comm_rank(load, &rank_new);
		if (rank_new == 0) // this is the root in the other partition 
		{
			is_second_root =true;
			MPI_Send(&proc_id, 1,MPI_INT, ROOT, DUMMY_TAG,MPI_COMM_WORLD);
			recvDataFromRoot(res,GPU_present);
		}
	}

	else{
		if(proc_id == ROOT){
			MPI_Recv(&second_root,1,MPI_INT,MPI_ANY_SOURCE,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sendDataToRoot(res,second_root,GPU_present);
		}
	}



	pair<int,int> partition_idx = get_partition_index(res.dim, root_partition); // to know what parts have we given to the other side
	int st_idx = partition_idx.first, en_idx = partition_idx.second;

	if (proc_id == ROOT) {
		if (st_idx == 0) out_local = &out[en_idx+1];
		else out_local = out;
	}

	else if (is_second_root) {
		out_local = (lll*)malloc(sizeof(lll)*(en_idx - st_idx+1));
	}

	

	sendDataToGroup(res, GPU_present, out_local, proc_id, load);

	// /* this is the root */
	if (proc_id == ROOT) {
		// need to receive from the other
		MPI_Recv(&out[st_idx] ,en_idx - st_idx + 1,MPI_LONG_LONG,MPI_ANY_SOURCE,DUMMY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	}

	else if (is_second_root) {
		MPI_Send(out_local, en_idx - st_idx + 1,MPI_LONG_LONG, ROOT, DUMMY_TAG,MPI_COMM_WORLD);
	
	}
}

//split equal number of rows
//pass complete vector
void perform_dot(Result& res, lll* out){
    int proc_id,nproc;
    
    int GPU_present;
    
	//MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_id);
    
    cudaError_t error = cudaGetDeviceCount(&GPU_present);	
	GPU_present = error == cudaSuccess ;

	int gpu_count;
	MPI_Allreduce(&GPU_present, &gpu_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	//printf("%d\n", gpu_count);
	if (gpu_count == nproc) {

		if (!optimize_gpu_usage || nproc <= gpu_procs){
			sendDataToGroup(res, 1, out, proc_id, MPI_COMM_WORLD);
			return;
		}

		GPU_present = proc_id < gpu_procs; 
		
		partition_and_compute(GPU_present, proc_id, gpu_procs, res, out);
		
	}

	else if (gpu_count == 0) {
		sendDataToGroup(res, 0, out, proc_id, MPI_COMM_WORLD);
		return;
	}

	else partition_and_compute(GPU_present, proc_id, gpu_count, res, out);
}   



int main(int argc,char* argv[]){
    int nproc;
    int id;

    //MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if(argc!=6){
        cout<< "usage: " <<argv[0]<< " <input_file>  <output_file> <inspect_id>  <num-streams> <gpu_procs>\n";
		MPI_Finalize();
        return -1;
    }


    Result res;
    lll* product;
    char* fname = argv[1];
	inspect_id= atoi(argv[3]);
   use_streams = atoi(argv[4]);
   gpu_procs = atoi(argv[5]);

	optimize_gpu_usage = (gpu_procs > 0);

    if(id == ROOT){
        res = read_data(fname);
        product = (lll*)malloc(res.dim*sizeof(lll));   
		for (int i=0; i < res.dim; i++) product[i] = 0;
    }
    
    perform_dot(res,product);

    if (id == ROOT){
	 write_data(product, res.dim, argv[2]); 
 	}

    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

