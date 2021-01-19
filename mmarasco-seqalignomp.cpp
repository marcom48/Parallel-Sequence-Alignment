// CPP program to solve the sequence alignment
// problem. Adapted from https://www.geeksforgeeks.org/sequence-alignment-problem/ and
// fixed an error when initializing the dp array :-)
#include <sys/time.h>
#include <string>
#include <cstring>
#include <iostream>

using namespace std;

int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap, int* xans, int* yans);


// Return current time, for performance measurement
uint64_t GetTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
}


// Driver code
int main(){
	int misMatchPenalty;
	int gapPenalty;
	std::string gene1;
	std::string gene2;
	std::cin >> misMatchPenalty;
	std::cin >> gapPenalty;
	std::cin >> gene1;
	std::cin >> gene2;
	std::cout << "misMatchPenalty=" << misMatchPenalty << std::endl;
	std::cout << "gapPenalty=" << gapPenalty << std::endl;

	int m = gene1.length(); // length of gene1
	int n = gene2.length(); // length of gene2
	int l = m+n;
	int xans[l+1], yans[l+1];

	uint64_t start = GetTimeStamp ();

	// calling the function to calculate the result
	int penalty = getMinimumPenalty(gene1, gene2,
		misMatchPenalty, gapPenalty,
		xans,yans);
	
	// print the time taken to do the computation
	printf("Time: %ld us\n", (uint64_t) (GetTimeStamp() - start));
	
	// postprocessing of the answer, for printing results

	// Since we have assumed the answer to be n+m long,
	// we need to remove the extra gaps in the starting
	// id represents the index from which the arrays
	// xans, yans are useful
	int id = 1;
	int i;
	for (i = l; i >= 1; i--)
	{
		if ((char)yans[i] == '_' && (char)xans[i] == '_')
		{
			id = i + 1;
			break;
		}
	}
	
	// Printing the final answer
	std::cout << "Minimum Penalty in aligning the genes = ";
	std::cout << penalty << std::endl;
	std::cout << "The aligned genes are :" << std::endl;
	for (i = id; i <= l; i++)
	{
		std::cout<<(char)xans[i];
	}
	std::cout << "\n";
	for (i = id; i <= l; i++)
	{
		std::cout << (char)yans[i];
	}
	std::cout << "\n";

	return 0;
}
int min3(int a, int b, int c) {
	if (a <= b && a <= c) {
		return a;    
	} else if (b <= a && b <= c) {
		return b;
	} else {
		return c;    
	}
}

/******************************************************************************/
/* Do not change any lines above here.            */
/* All of your changes should be below this line. */
/******************************************************************************/

/******************************************************************************

COMP90025 Parallel and Multicore Computing Project 2A 2020

Marco Marasco (834482)

This file contains a parallelised dynamic programming approach to solving
the Sequence Alignment problem  - 
https://www.geeksforgeeks.org/sequence-alignment-problem/.

The sequence alignment problem is roughly, given two sequences of symbols,
insert gaps into the sequences such that the penalty, that is a function of
how many gaps are inserted and mis-matches in symbol alignment, is minimised,
when the two sequences are compared symbol for symbol at each index location.

******************************************************************************/
#include <omp.h>

// Chunk size.
#define TILE 1000

// equivalent of  int *dp[width] = new int[height][width]
// but works for width not known at compile time.
// (Delete structure by  delete[] dp[0]; delete[] dp;)
unsigned **new2d (int width, int height)
{
	unsigned **dp = new unsigned *[width];
	size_t size = width;
	size *= height;
	unsigned *dp0 = new unsigned [size];
	if (!dp || !dp0)
	{
	    std::cerr << "getMinimumPenalty: new failed" << std::endl;
	    exit(1);
	}
	dp[0] = dp0;
	for (int i = 1; i < width; i++)
	    dp[i] = dp[i-1] + height;

	return dp;
}



/*
Function computes and returns the minimum penalty for alignment.
*/
int getMinimumPenalty(std::string x, std::string y, int pxy, int pgap,
	int* xans, int* yans)
{

	// Iterator related variables
	unsigned ii, i, j, k, l, diag_length, xpos, ypos, first;
	
	unsigned m = x.length(); // length of gene1
	unsigned n = y.length(); // length of gene2

	// Note: Testing showed jj as int to be faster.
	int jj, ret;

	// Integer value of an '_' character.
	int under = (int)'_';

	// Initialise dynamic programming table.
	unsigned **dp = new2d(m + 1, n + 1);


	// Initialise table values.
	unsigned smaller = m < n ? m : n;
	#pragma omp parallel for simd shared(dp, pgap) linear(i)
	for (i = 0; i <= smaller; i++)
	{
		dp[i][0] = i * pgap;
		dp[0][i] = i * pgap;
	}

	// Assume |m - n| is computationally small - don't parallelise.
	if (smaller == m){
		for (i = smaller + 1; i <= n; i++)
		{
			dp[0][i] = i * pgap;
		}
	} else{
		for (i = smaller + 1; i <= m; i++)
		{
			dp[i][0] = i * pgap;
		}
	}

	
	// The DP algorithm for position (i,j) in the table has dependencies on
	// positions (i - 1, j - 1), (i - 1, j), (i, j - 1). Therefore, approach
	// is to calculate each antidiagonal (top right to bottom left) of table
	// in parallel. To reduce thread overheads, rather than every thread
	// computing a single table value then being reassigned, each thread
	// is assigned a TILE * TILE chunk of the table to compute.

	// Below approach starts antidiagonals from top left, and moves 
	// right (and down if required) across the table for each antidiagonal
	// section of tiles.
	
	for (i = 0, j = 0; j <= n + TILE; j+= TILE)
	{
		
		// Number of tiles in antidiagonal to be computed.
		diag_length = j < m - i ? 1 + (j/TILE) : 1 + (m - i)/TILE;


		#pragma omp parallel for num_threads(diag_length) schedule(dynamic) shared (x, y, dp, pxy, pgap, m, n) private(ii, jj, k, first)
		for(k = 0; k < diag_length; k++){
			
			// Iterate over thread assigned tile.
			for(ii = i + k*TILE + 1;  ii <= m && ii < i + k*TILE + 1 + TILE; ii++){
				for(jj = j - k*TILE + 1; jj <=  n &&  jj < j - k*TILE + 1 + TILE; jj++){
					
					// Compute DP value.
					if (x[ii - 1] == y[jj - 1])
					{
						dp[ii][jj] = dp[ii - 1][jj - 1];
						continue;
					}
					else
					{

						// first =  min of (dp[ii-1][jj-1] + pxy, dp[ii - 1][jj] + pgap)  
						first = dp[ii-1][jj-1] + pxy <= dp[ii - 1][jj] + pgap ? dp[ii-1][jj-1] + pxy : dp[ii - 1][jj] + pgap;

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
			
			if  (i > m + TILE){
				break;
			} else{
				i+= TILE;

			};
		}
	}
	
		

	// Reconstructing the solution
	l = n + m; // maximum possible length

	i = m;
	j = n;

	xpos = l;
	ypos = l;
	

	while ( !(i == 0 || j == 0))
	{
		if (x[i - 1] == y[j - 1])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)y[j - 1];
			i--; j--;
		}
		else if (dp[i - 1][j - 1] + pxy == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = (int)y[j - 1];
			i--; j--;
		}
		else if (dp[i - 1][j] + pgap == dp[i][j])
		{
			xans[xpos--] = (int)x[i - 1];
			yans[ypos--] = under;
			i--;
		}
		else if (dp[i][j - 1] + pgap == dp[i][j])
		{
			xans[xpos--] = under;
			yans[ypos--] = (int)y[j - 1];
			j--;
		}
	}
	while (xpos > 0)
	{
		if (i > 0) xans[xpos--] = (int)x[--i];
		else xans[xpos--] = under;
	}
	while (ypos > 0)
	{
		if (j > 0) yans[ypos--] = (int)y[--j];
		else yans[ypos--] = under;
	}

	ret = dp[m][n];

	return ret;
}


// g++ -fopenmp -o mmarasco-seqalignomp -O3 mmarasco-seqalignomp.cpp