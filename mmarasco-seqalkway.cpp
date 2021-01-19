// CPP program to solve the sequence alignment
// problem. Adapted from https://www.geeksforgeeks.org/sequence-alignment-problem/ 
// with many fixes and changes for multiple sequence alignment and to include an MPI driver
#include <mpi.h>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <iostream>
#include "sha512.hh"

using namespace std;

std::string getMinimumPenalties(std::string *genes, int k, int pxy, int pgap, int *penalties);
int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int *xans, int *yans);
void do_MPI_task(int rank);

/*
Examples of sha512 which returns a std::string
sw::sha512::calculate("SHA512 of std::string") // hash of a string, or
sw::sha512::file(path) // hash of a file specified by its path, or
sw::sha512::calculate(&data, sizeof(data)) // hash of any block of data
*/

// Return current time, for performance measurement
uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}

const MPI_Comm comm = MPI_COMM_WORLD;
const int root = 0;

// Driver code
int main(int argc, char **argv){
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &rank);
	if(rank==root){
		int misMatchPenalty;
		int gapPenalty;
		int k;
		std::cin >> misMatchPenalty;
		std::cin >> gapPenalty;
		std::cin >> k;	
		std::string genes[k];
		for(int i=0;i<k;i++) std::cin >> genes[i];

		int numPairs= k*(k-1)/2;

		int penalties[numPairs];
		
		uint64_t start = GetTimeStamp ();

		// return all the penalties and the hash of all allignments
		std::string alignmentHash = getMinimumPenalties(genes,
			k,misMatchPenalty, gapPenalty,
			penalties);
		
		// print the time taken to do the computation
		printf("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));
		
		// print the alginment hash
		std::cout<<alignmentHash<<std::endl;

		for(int i=0;i<numPairs;i++){
			std::cout<<penalties[i] << " ";
		}
		std::cout << std::endl;
	} else {
		// do stuff for MPI tasks that are not rank==root
		do_MPI_task(rank);
	}
	MPI_Finalize();
	return 0;
}

/******************************************************************************/
/* Do not change any lines above here.            */
/* All of your changes should be below this line. */
/******************************************************************************/
/******************************************************************************

COMP90025 Parallel and Multicore Computing Project 2B 2020

Marco Marasco (834482)

This file contains a distributed parallelised dynamic programming approach
to solving the Sequence Alignment problem for k different genes, using
OpenMP and OpenMPI.

The approach is to distribute jobs to each worker, who will compute the
results, and then return them to the root node to compute the final values
required for output.

******************************************************************************/
#include <omp.h>
#define HASH_BUFF 257
#define TILE 1000
void mygetMinimumPenalty(unsigned** dp, char* x, char* y, int pxy, int pgap,
						char *xans, char *yans, int m, int n);
void mygetMinimumPenaltyString(unsigned** dp, std::string x, std::string y, int pxy, int pgap,
						char *xans, char *yans, int m, int n);

// Stores job results.
struct computedType
{
	int probNum;
	int penalty;
	char hash[HASH_BUFF];
};


// equivalent of  int *dp[width] = new int[height][width]
// but works for width not known at compile time.
// (Delete structure by  delete[] dp[0]; delete[] dp;)
unsigned **mynew2d(int width, int height)
{
	unsigned **dp = new unsigned *[width];
	size_t size = width;
	size *= height;
	unsigned *dp0 = new unsigned[size];
	if (!dp || !dp0)
	{
		std::cerr << "getMinimumPenalty: new failed" << std::endl;
		exit(1);
	}
	dp[0] = dp0;
	for (int i = 1; i < width; i++)
		dp[i] = dp[i - 1] + height;

	return dp;
}


// called by the root MPI task only
// this procedure should distribute work to other MPI tasks
// and put together results, etc.
std::string getMinimumPenalties(std::string *genes, int k, int pxy, int pgap,
								int *penalties)
{
	// Initialise variables.
	int rank, size, pairs, maxLen, i, j, l, m, n, completed_jobs, currprob, id, a, probNum;

	// Calculate rank and size.
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);


	// Find max gene length and total summed length.
	maxLen = -1;
	int total_len = 0;
	for (i = 0; i < k; i++)
	{
		if ((int)genes[i].length() > maxLen)
		{
			maxLen = (int)genes[i].length();
		}
		total_len += (int)genes[i].length();
	}

	// Safety barrier.
	maxLen += 1;

	// Broadcast number of genes, maxLen, pxy, pgap, pairs.
	pairs = (k * (k - 1)) / 2;
	int vals[5] = {k, maxLen, pxy, pgap, pairs};
	MPI_Bcast(vals, 5, MPI_INT, root, comm);


	// Broadcast gene lengths.
	int gene_lens[k];
	for (i = 0; i < k; i++)
	{
		gene_lens[i] = (int)genes[i].length();
	}
	MPI_Bcast(gene_lens, k, MPI_INT, root, comm);

	// Calculate prefix sum of genes.
	int gene_prefixes[k] = {0};
	for(i = 1; i < k; i++){
		gene_prefixes[i] = gene_prefixes[i-1] +  gene_lens[i-1] + 1;
	}

	// Broadcast genes.
	char b_genes[(size_t) (total_len + k)];
	for(i = 0; i < k; i++){
		strcpy(&b_genes[gene_prefixes[i]], genes[i].c_str());
	}
	MPI_Bcast(b_genes, total_len + k, MPI_CHAR, root, comm);


	// Create MPI_Type for results from workers.
	MPI_Datatype computedType;
	int lengths[3] = {1, 1, HASH_BUFF};
	const MPI_Aint displacements[3] = {0, sizeof(int), 2 * sizeof(int)};
	MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_CHAR};
	MPI_Type_create_struct(3, lengths, displacements, types, &computedType);
	MPI_Type_commit(&computedType);


	// Calculate number of jobs to receive per worker
	// and the longest genes this node will compute.
	int recv_count[size] = {0};

	// Max len of first gene passed to penalty function
	int my_max_i = -1;
	
	// Max len of second gene passed to penalty function
	int my_max_j = -1;
	currprob = 0;
	for (i = 1; i < k; i++)
	{
		for (j = 0; j < i; j++)
		{
			// Check if worker should do job.
			if ((1 + currprob) % size == 0)
			{	
				// Update max lengths.
				my_max_i = gene_lens[i] > my_max_i ? gene_lens[i] : my_max_i;
				my_max_j = gene_lens[j] > my_max_j ? gene_lens[j] : my_max_j;
			}
			
			recv_count[(1 + currprob) % size] += 1;
			currprob += 1;
		}
	}


	// Results for computations
	struct computedType results[recv_count[0]];
	completed_jobs = 0;
	probNum = 0;

	// Maximum gene length for this worker.
	int my_max = my_max_i > my_max_j ? my_max_i : my_max_j;

	// Penalty buffer arrays.
	char xans[(2 * my_max) + 2], yans[(2 * my_max) + 2];
	

	// DP table as large as required for this node (with a little safety).
	unsigned **dp = mynew2d(my_max_i + 2, my_max_j + 2);

	// Initialise dp array.
	unsigned smaller = my_max_i < my_max_j ? my_max_i : my_max_j;
	#pragma omp parallel for simd shared(dp, pgap) linear(i)
	for (i = 0; i <= smaller + 1; i++)
	{
		dp[i][0] = i * pgap;
		dp[0][i] = i * pgap;
	}

	if (smaller == my_max_i)
	{
		for (i = smaller + 1; i <= my_max_j + 1; i++)
		{
			dp[0][i] = i * pgap;
		}
	}
	else
	{
		for (i = smaller + 1; i <= my_max_i+1; i++)
		{
			dp[i][0] = i * pgap;
		}
	}

	std::string align1;
	std::string align2;
	for (i = 1; i < k; i++)
	{
		for (j = 0; j < i; j++)
		{

			// Check if worker should do job.
			if ((1 + probNum) % size != 0)
			{
				probNum += 1;
				continue;
			}

			m = genes[i].length(); // length of gene1
			n = genes[j].length(); // length of gene2
			l = m + n;


			// Compute penalty.
			mygetMinimumPenaltyString(dp, genes[i], genes[j], pxy, pgap, xans, yans,m,n);

			// Since we have assumed the answer to be n+m long,
			// we need to remove the extra gaps in the starting
			// id represents the index from which the arrays
			// xans, yans are useful
			id = 1;

			for (a = l; a >= 1; a--)
			{
				if (yans[a] == '_' && xans[a] == '_')
				{
					id = a + 1;
					break;
				}
			}

			align1.reserve(id+l);
			align2.reserve(id+l);

			for (a = id; a <= l; a++)
			{
				align1 += xans[a];
				align2 += yans[a];

			}
		
		
			// Store results.
			memcpy(results[completed_jobs].hash, sw::sha512::calculate(sw::sha512::calculate(align1).append(sw::sha512::calculate(align2))).c_str(), HASH_BUFF);
			results[completed_jobs].probNum = probNum;
			results[completed_jobs].penalty = (int)dp[m][n];

			// Update number of jobs worker has done.
			completed_jobs += 1;
			
			align1.clear();
			align2.clear();			

			probNum++;
		}
	}


	// Receive buffer.
	struct computedType recvbuffer[pairs];

	// Displacement of messages for each process inside recvbuffer.
	int disps[size];
	disps[0] = 0;

	for (int i = 1; i < size; i++)
	{
		disps[i] = disps[i - 1] + recv_count[i - 1];
	}

	// Gather all results.
	MPI_Gatherv(results, recv_count[0], computedType, recvbuffer, recv_count, disps, computedType, root, comm);

	// Store results in correct problem number order.
	std::string collated[pairs];
	for (int i = 0; i < pairs; i++)
	{
		collated[recvbuffer[i].probNum] = recvbuffer[i].hash;
		penalties[recvbuffer[i].probNum] = recvbuffer[i].penalty;
	}

	std::string alignmentHash = "";

	// Comptute alignment hash sequentially.
	for (int i = 0; i < pairs; i++)
	{
		alignmentHash = sw::sha512::calculate(alignmentHash += collated[i]);
	}

	return alignmentHash;
}

// called for all tasks with rank!=root
// do stuff for each MPI task based on rank
void do_MPI_task(int rank)
{

	// Intialise variable.
	int size, i, j, m, n, l, id, a, jobs_to_compute, completed_jobs, probNum, penalty;

	MPI_Comm_size(comm, &size);


	// Initialise return type.
	MPI_Datatype computedType;
	int lengths[3] = {1, 1, HASH_BUFF};
	const MPI_Aint displacements[3] = {0, sizeof(int), 2 * sizeof(int)};
	MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_CHAR};
	MPI_Type_create_struct(3, lengths, displacements, types, &computedType);
	MPI_Type_commit(&computedType);

	// Receive values k, max_gene_length, pxy, pgap, pairs.
	int vals[5];
	MPI_Bcast(vals, 5, MPI_INT, root, comm);

	// Stores length of genes
	int gene_lens[vals[0]];

	// Get gene lengths.
	MPI_Bcast(gene_lens, vals[0], MPI_INT, root, comm);

	// Compute gene length prefix.
	int gene_prefixes[vals[0]];
	gene_prefixes[0] = 0;
	int total_len = gene_lens[0];
	for(i = 1; i < vals[0]; i++){
		total_len += gene_lens[i];
		gene_prefixes[i] = gene_prefixes[i-1] +  gene_lens[i-1] + 1;
	}

	// Receive genes.
	char b_genes[(size_t) (total_len + vals[0])];
	MPI_Bcast(b_genes, total_len+vals[0], MPI_CHAR, root, comm);

	

	// Buffer to store results.
	jobs_to_compute = (vals[4] / size) + ((rank + 1) % size) + 1;
	struct computedType results[jobs_to_compute];
	

	// Find largest gene worker will compute.
	int my_max_i = -1;
	int my_max_j = -1;
	int my_max = 0;
	probNum = 0;
	for (i = 1; i < vals[0]; i++)
	{
		for (j = 0; j < i; j++)
		{
			if ((1 + probNum) % size == rank)
			{
				my_max_i = gene_lens[i] > my_max_i ? gene_lens[i] : my_max_i;
				my_max_j = gene_lens[j] > my_max_j ? gene_lens[j] : my_max_j;
				
			}
			probNum += 1;
			
		}
	}


	// Longest gene worker will compute.
	my_max = my_max_i > my_max_j ? my_max_i : my_max_j;


	// Penalty answer arrays.
	char xans[(2 * my_max) + 2], yans[(2 * my_max) + 2];

	// Initialise dp table
	unsigned **dp = mynew2d(my_max_i + 2, my_max_j + 2);
	int pgap = vals[3];
	unsigned smaller = my_max_i < my_max_j ? my_max_i : my_max_j;
	#pragma omp parallel for simd shared(dp, pgap) linear(i)
	for (i = 0; i <= smaller + 1; i++)
	{
		dp[i][0] = i * pgap;
		dp[0][i] = i * pgap;
	}

	if (smaller == my_max_i)
	{
		for (i = smaller + 1; i <= my_max_j + 1; i++)
		{
			dp[0][i] = i * pgap;
		}
	}
	else
	{
		for (i = smaller + 1; i <= my_max_i+1; i++)
		{
			dp[i][0] = i * pgap;
		}
	}


	probNum = 0;
	completed_jobs = 0;
	std::string align1 = "";
	std::string align2 = "";

	
	for (i = 1; i < vals[0]; i++)
	{
		for (j = 0; j < i; j++)
		{
			// Check if worker should do job.
			if ((1 + probNum) % size != rank)
			{
				probNum += 1;
				continue;
			}

			m = gene_lens[i]; // length of gene1
			n = gene_lens[j]; // length of gene2
			l = m + n;

			// Compute penality
			mygetMinimumPenalty(dp, &b_genes[gene_prefixes[i]], &b_genes[gene_prefixes[j]], vals[2], vals[3], xans, yans,m,n);

			id = 1;

			for (a = l; a >= 1; a--)
			{
				if (yans[a] == '_' && xans[a] == '_')
				{
					id = a + 1;
					break;
				}
			}

			align1.reserve(id+l);
			align2.reserve(id+l);


			// Assuming this won't be very large - no need to parallelise.			
			for (a = id; a <= l; a++)
			{
				align1 += xans[a];
				align2 += yans[a];
			}

			// Store results.
			memcpy(results[completed_jobs].hash, sw::sha512::calculate(sw::sha512::calculate(align1).append(sw::sha512::calculate(align2))).c_str(), HASH_BUFF);
			results[completed_jobs].probNum = probNum;
			results[completed_jobs].penalty = (int)dp[m][n];

			completed_jobs++;

			align1.clear();
			align2.clear();


			probNum++;
		}
	}

	// Send results to master node.
	MPI_Gatherv(results, completed_jobs, computedType, NULL, NULL, NULL, NULL, root, comm);
	

}


/*
Function computes and returns the minimum penalty for alignment.
Accepts char* objects as genes.
*/
void mygetMinimumPenalty(unsigned** dp, char* x, char* y, int pxy, int pgap,
						char *xans, char *yans, int m, int n)
{
	// Iterator related variables
	unsigned ii, i, j, k, l, diag_length, xpos, ypos, first;

	// unsigned m = x.length(); // length of gene1
	// unsigned n = y.length(); // length of gene2

	// Note: Testing showed jj as int to be faster.
	int jj, ret;

	// The DP algorithm for position (i,j) in the table has dependencies on
	// positions (i - 1, j - 1), (i - 1, j), (i, j - 1). Therefore, approach
	// is to calculate each antidiagonal (top right to bottom left) of table
	// in parallel. To reduce thread overheads, rather than every thread
	// computing a single table value then being reassigned, each thread
	// is assigned a TILE * TILE chunk of the table to compute.

	// Below approach starts antidiagonals from top left, and moves
	// right (and down if required) across the table for each antidiagonal
	// section of tiles.

	for (i = 0, j = 0; j <= n + TILE; j += TILE)
	{

		// Number of tiles in antidiagonal to be computed.
		diag_length = j < m - i ? 1 + (j / TILE) : 1 + (m - i) / TILE;

#pragma omp parallel for num_threads(diag_length) schedule(dynamic) shared(x, y, dp, pxy, pgap, m, n) private(ii, jj, k, first)
		for (k = 0; k < diag_length; k++)
		{
			// std::cout << "Calling in from openmp rank " << omp_get_thread_num() << std::endl;
			// Iterate over thread assigned tile.
			for (ii = i + k * TILE + 1; ii <= m && ii < i + k * TILE + 1 + TILE; ii++)
			{
				for (jj = j - k * TILE + 1; jj <= n && jj < j - k * TILE + 1 + TILE; jj++)
				{

					// Compute DP value.
					if (x[ii - 1] == y[jj - 1])
					{
						dp[ii][jj] = dp[ii - 1][jj - 1];
						continue;
					}
					else
					{

						// first =  min of (dp[ii-1][jj-1] + pxy, dp[ii - 1][jj] + pgap)
						first = dp[ii - 1][jj - 1] + pxy <= dp[ii - 1][jj] + pgap ? dp[ii - 1][jj - 1] + pxy : dp[ii - 1][jj] + pgap;

						dp[ii][jj] = dp[ii][jj - 1] + pgap <= first ? dp[ii][jj - 1] + pgap : first;

						continue;
					}
				}
			}
		}

		// Update starting position of antidiagonal.
		if (j >= n)
		{
			j = n - TILE;

			if (i > m + TILE)
			{
				break;
			}
			else
			{
				i += TILE;
			};
		}
	}

	// Reconstructing the solution
	l = n + m; // maximum possible length

	i = m;
	j = n;

	xpos = l;
	ypos = l;

	while (!(i == 0 || j == 0))
	{
		if (x[i - 1] == y[j - 1])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = y[j - 1];
			i--;
			j--;
		}
		else if (dp[i - 1][j - 1] + pxy == dp[i][j])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = y[j - 1];
			i--;
			j--;
		}
		else if (dp[i - 1][j] + pgap == dp[i][j])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = '_';
			i--;
		}
		else if (dp[i][j - 1] + pgap == dp[i][j])
		{
			xans[xpos--] = '_';
			yans[ypos--] = y[j - 1];
			j--;
		}
	}
	while (xpos > 0)
	{
		if (i > 0)
			xans[xpos--] = x[--i];
		else
			xans[xpos--] = '_';
	}
	while (ypos > 0)
	{
		if (j > 0)
			yans[ypos--] = y[--j];
		else
			yans[ypos--] = '_';
	}

}

/*
Function computes and returns the minimum penalty for alignment.
Accepts std::string objects for genes.
*/
void mygetMinimumPenaltyString(unsigned** dp, std::string x, std::string y, int pxy, int pgap,
						char *xans, char *yans, int m, int n)
{
	// Iterator related variables
	unsigned ii, i, j, k, l, diag_length, xpos, ypos, first;

	// unsigned m = x.length(); // length of gene1
	// unsigned n = y.length(); // length of gene2

	// Note: Testing showed jj as int to be faster.
	int jj, ret;

	// The DP algorithm for position (i,j) in the table has dependencies on
	// positions (i - 1, j - 1), (i - 1, j), (i, j - 1). Therefore, approach
	// is to calculate each antidiagonal (top right to bottom left) of table
	// in parallel. To reduce thread overheads, rather than every thread
	// computing a single table value then being reassigned, each thread
	// is assigned a TILE * TILE chunk of the table to compute.

	// Below approach starts antidiagonals from top left, and moves
	// right (and down if required) across the table for each antidiagonal
	// section of tiles.

	for (i = 0, j = 0; j <= n + TILE; j += TILE)
	{

		// Number of tiles in antidiagonal to be computed.
		diag_length = j < m - i ? 1 + (j / TILE) : 1 + (m - i) / TILE;

#pragma omp parallel for num_threads(diag_length) schedule(dynamic) shared(x, y, dp, pxy, pgap, m, n) private(ii, jj, k, first)
		for (k = 0; k < diag_length; k++)
		{
			// std::cout << "Calling in from openmp rank " << omp_get_thread_num() << std::endl;
			// Iterate over thread assigned tile.
			for (ii = i + k * TILE + 1; ii <= m && ii < i + k * TILE + 1 + TILE; ii++)
			{
				for (jj = j - k * TILE + 1; jj <= n && jj < j - k * TILE + 1 + TILE; jj++)
				{

					// Compute DP value.
					if (x[ii - 1] == y[jj - 1])
					{
						dp[ii][jj] = dp[ii - 1][jj - 1];
						continue;
					}
					else
					{

						// first =  min of (dp[ii-1][jj-1] + pxy, dp[ii - 1][jj] + pgap)
						first = dp[ii - 1][jj - 1] + pxy <= dp[ii - 1][jj] + pgap ? dp[ii - 1][jj - 1] + pxy : dp[ii - 1][jj] + pgap;

						dp[ii][jj] = dp[ii][jj - 1] + pgap <= first ? dp[ii][jj - 1] + pgap : first;

						continue;
					}
				}
			}
		}

		// Update starting position of antidiagonal.
		if (j >= n)
		{
			j = n - TILE;

			if (i > m + TILE)
			{
				break;
			}
			else
			{
				i += TILE;
			};
		}
	}

	// Reconstructing the solution
	l = n + m; // maximum possible length

	i = m;
	j = n;

	xpos = l;
	ypos = l;

	while (!(i == 0 || j == 0))
	{
		if (x[i - 1] == y[j - 1])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = y[j - 1];
			i--;
			j--;
		}
		else if (dp[i - 1][j - 1] + pxy == dp[i][j])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = y[j - 1];
			i--;
			j--;
		}
		else if (dp[i - 1][j] + pgap == dp[i][j])
		{
			xans[xpos--] = x[i - 1];
			yans[ypos--] = '_';
			i--;
		}
		else if (dp[i][j - 1] + pgap == dp[i][j])
		{
			xans[xpos--] = '_';
			yans[ypos--] = y[j - 1];
			j--;
		}
	}
	while (xpos > 0)
	{
		if (i > 0)
			xans[xpos--] = x[--i];
		else
			xans[xpos--] = '_';
	}
	while (ypos > 0)
	{
		if (j > 0)
			yans[ypos--] = y[--j];
		else
			yans[ypos--] = '_';
	}

}


// mpicxx -o mmarasco-seqalkway mmarasco-seqalkway.cpp -fopenmp -O3