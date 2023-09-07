// Owen Lucas
// Based on code from Kyle Buettner
// Two bit counting bloom filter
// Uses Algorithm from BFCounter and NEST
// IMPORTANT: Consider atomic updates and race conditions when parallelizing


//Remove the entire first loop and see what the throughput looks like

#include <omp.h>			// Included for timing
#include <string>			
#include <iostream>
#include <bits/stdc++.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include "fasta.c"

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <CL/sycl.hpp>
using namespace cl::sycl;

#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

#include "dpc_common.hpp"

#define MAX_SEQ_L 100

// Key parameters for Bloom filter -> These determine false positive probability (along with N)
// M is now the number of indecies in each partition of the filter
#define M 0x200000000
#define K 31
#define MAX_K_BYTES 8

#define H 2
#define BANKS 16
#define BF_BANKS 15

#define BIG_CONSTANT(x) (x##LLU)

#define NSEC_IN_SEC 1000000000.0


class produce_reads1;
class parse_reads1;
class add_to_bf;

class produce_reads2;
class parse_reads2;
class query_bf;

class compile_query;
        

struct sequence {
    std::bitset<2*MAX_SEQ_L> seq = 0;
    unsigned int             L = 0;
    bool                     valid = false;
	//bool					 problem = false; //used for debugging
};

struct query {
    //std::bitset<MAX_SEQ_L - K + 1> found = 0;
	std::bitset<MAX_SEQ_L - K + 1> seq = 0;
	//std::bitset<2*MAX_SEQ_L> seq = 0;
    int seen = 0;
};

struct hash_q {
    unsigned int results[H];
    unsigned int j = 0;
    unsigned int i = 0;
};

struct kmer_data{
    std::bitset<MAX_K_BYTES * 8> kmer;
    unsigned int seq_num = 0; // What sequence does this data go with
    unsigned short kmer_num = 0; // What kmer in the sequence does it go with

};

struct hash_result{
    unsigned long long int result = 0;
    bool done = false;
    unsigned int seq_num = 0; // What sequence does this data go with
    unsigned short kmer_num = 0; // What kmer in the sequence does it go with
    unsigned short hash_num = 0; // What hash function is it from
};

struct bank_query{
    bool found;
    unsigned int seq_num = 0; // What sequence does this data go with
    unsigned short kmer_num = 0; // What kmer in the sequence does it go with
};

struct partial_query {
    std::bitset<MAX_SEQ_L - K + 1> found = 0;
	//std::bitset<2*MAX_SEQ_L> found = 0;
    int seq_num = 0;
	//bool problem = false; //used for debugging
};

using seq_pipe = ext::intel::pipe<class some_pipe, sequence, 128>;
//using seq_pipe2 = ext::intel::pipe<class other_pipe, sequence, 128>;
using res_pipe = ext::intel::pipe<class other_pipe2, unsigned long long int, 256>;
//using res_pipe2 = ext::intel::pipe<class other_pipe3, hash_q, 16>;
//using kmer2hash = ext::intel::pipe<class other_pipe4, kmer_data, 16>;
//using hash2bank = ext::intel::pipe<class other_pipe5, hash_result, 16>;
//using bank2comp = ext::intel::pipe<class other_pipe5, bank_query, 16>;

using kmer2hash_pipes = fpga_tools::PipeArray<    // Defined in "pipe_utils.hpp".
    class array_pipe1,               // An identifier for the pipe.
    kmer_data,   // The type of data in the pipe.
    256,                            // The capacity of each pipe.
    H                             // array dimension.
    >;

using hash2bank_pipes = fpga_tools::PipeArray<    // Defined in "pipe_utils.hpp".
    class array_pipe2,               // An identifier for the pipe.
    hash_result,                    // The type of data in the pipe.
    256,                            // The capacity of each pipe.
    (BF_BANKS)                         // array dimension.
    >;
	
using bank2comp_pipes = fpga_tools::PipeArray<    // Defined in "pipe_utils.hpp".
    class array_pipe3,               // An identifier for the pipe.
    bank_query,                    // The type of data in the pipe.
    256,                            // The capacity of each pipe.
    (BF_BANKS)                         // array dimension.
    >;
	
using len2comp_pipes = fpga_tools::PipeArray<    // Defined in "pipe_utils.hpp".
    class array_pipe4,               // An identifier for the pipe.
    int,   // The type of data in the pipe.
    2048,                            // The capacity of each pipe.
    H                             // array dimension.
    >;
	
using comp2final_pipes = fpga_tools::PipeArray<    // Defined in "pipe_utils.hpp".
    class array_pipe5,               // An identifier for the pipe.
    partial_query,   // The type of data in the pipe.
    256,                            // The capacity of each pipe.
    H                             // array dimension.
    >;




unsigned int MurmurHash2 ( const void * key, unsigned int seed )
{
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.

	const unsigned int m = 0x5bd1e995;
	const int r = 24;
    const int length =  MAX_K_BYTES; 
	// Initialize the hash to a 'random' value

	unsigned int h = seed ^ length;

	// Mix 4 bytes at a time into the hash

	const unsigned char * data = (const unsigned char *)key;

	//while(len >= 4)
    for (int x = 4; x <= length; x = x + 4)
	{
		unsigned int k = *(unsigned int *)data;

		k *= m; 
		k ^= k >> r; 
		k *= m; 
		
		h *= m; 
		h ^= k;

		data += 4;
		//len -= 4;
	}
	
	// Handle the last few bytes of the input array
    
    int rem = length & 3;
    
    if (rem != 0){

        switch(rem)
        {
        case 3: h ^= data[2] << 16;
        case 2: h ^= data[1] << 8;
        case 1: h ^= data[0];
                h *= m;
        };
    }

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.

	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}

unsigned long long int MurmurHash64A ( const void * key, uint64_t seed )
{
  const long long unsigned int m = BIG_CONSTANT(0xc6a4a7935bd1e995);
  const int r = 47;
  const int length =  MAX_K_BYTES; 

  unsigned long long int h = seed ^ (length * m);

  const unsigned long long int * data = (const long long unsigned int *)key;
  const unsigned long long int * end = data + (length/8);

   for (int x = 8; x <= length; x = x + 8)
  {
    long long unsigned int k = *data++;

    k *= m; 
    k ^= k >> r; 
    k *= m; 
    
    h ^= k;
    h *= m; 
  }

  const unsigned char * data2 = (const unsigned char*)data;
    
  int rem = length & 7;
  if(rem != 0){

      switch(rem)
      {
          case 7: h ^= ((long long unsigned int) data2[6]) << 48;
          case 6: h ^= ((long long unsigned int) data2[5]) << 40;
          case 5: h ^= ((long long unsigned int) data2[4]) << 32;
          case 4: h ^= ((long long unsigned int) data2[3]) << 24;
          case 3: h ^= ((long long unsigned int) data2[2]) << 16;
          case 2: h ^= ((long long unsigned int) data2[1]) << 8;
          case 1: h ^= ((long long unsigned int) data2[0]);
                  h *= m;
      };
  }
 
  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
} 

// Bloom Filter Code
void add_to_bloom_filter(unsigned long long int kmer, volatile char *bloom_filter){
	#pragma unroll
    for(int i = 0; i < H; i++){
        unsigned int result = MurmurHash2(&kmer, i) & (M-1);
        short lower = (result & 3) << 1;
        unsigned int upper = result >> 2;
        char bf_byte = bloom_filter[upper];
        unsigned int idx = 1 << lower;
        if(bf_byte && idx != 0){
            bf_byte = bf_byte | (idx << 1);
        }
		bf_byte = bf_byte |  idx;
        bloom_filter[upper] = bloom_filter[upper] | bf_byte;
	}
}

// Bloom Filter Code
int query_bloom_filter(unsigned long long int kmer, volatile char *bloom_filter){ 
	std::bitset<H> found;
    found.reset();
    #pragma unroll
	for(int i = 0; i < H; i++){
        unsigned int result = MurmurHash2(&kmer, i) & (M-1);
        short lower = (result & 3) << 1;
        unsigned int upper = result >> 2;
        unsigned int idx = 1 << lower;
		found[i] = ((bloom_filter[upper] & (idx << 1)) != 0);
	}
	return (int)found.all();
}

char convert_int_to_base(unsigned int base) {
	if (base == 0) {
		return 'A';
	}
	else if (base == 1) {
		return 'C';
	}
	else if (base == 2) {
		return 'G';
	}
	else if (base == 3) {
		return 'T';
	}
	return 'N';
}

void convert_int_to_word(unsigned long long int word, char *temp_word) {
	unsigned long long int shift_mask = 3; // Bits = 11
	for (int i = 0; i < K; i++) {
		char c = convert_int_to_base((word & (shift_mask << (2 * (K-i-1)))) >> (2 * (K-i-1)));
		temp_word[i] = c;
	}
	temp_word[K] = '\0';
}



unsigned long long int convert_base_to_int(char base){
	if(base == 'A'){
		return 0;
	}
	else if(base == 'C'){
		return 1;
	}
	else if(base == 'G'){
		return 2;
	}
	else if(base == 'T'){
		return 3;
	}
	else {//error
		return 0;
	}
}

unsigned long long int convert_word_to_int(char word[K+1]){
	unsigned long long int word_as_int = 0;
	for (int i = 0; i < K; i++){
		word_as_int |= (convert_base_to_int(word[i]) << (2*i));
	}
	return word_as_int;
}


void print_hash_lookup_table(std::unordered_map<unsigned long long int, int>* hash_lookup_table){
	std::unordered_map<unsigned long long int, int>::iterator itr; 
	char kmer[K+1];
	kmer[K] = '\0';
	int total = 0;
	for(itr = hash_lookup_table->begin(); itr !=  hash_lookup_table->end(); itr++){
		convert_int_to_word(itr->first, kmer);
		printf("\n%s: ", kmer);
		printf("%d", itr->second);
		total += itr->second;
		
	}
	printf("\nTotal # of k-mers in the hash table: %d\n", total);
}

 

int main (int argc, char *argv[]) {
    
    
    // Select either:
    //  - the FPGA emulator device (CPU emulation of the FPGA)
    //  - the FPGA device (a real FPGA)
    #if defined(FPGA_EMULATOR)
        ext::intel::fpga_emulator_selector device_selector;
    #else
        ext::intel::fpga_selector device_selector;
    #endif
    
    try{

        // Need to load files(s)
        // Maybe add command line argument to determine if short read or database
        // Command line argument for name of file

        //FILE *query_file;
        FILE *database_file;
	
        FASTAFILE *ffp;
        
        char *seq;
        char *name;
        int L;

        //char* file_name = "./data/sra_data.fasta";
		char* file_name = "./data/SRR034949.fasta";

        ffp = OpenFASTA(file_name);
        long long unsigned int num_database_char = 0;

        char c;

        char kmer[K+1];
        kmer[K] = '\0';
        long long unsigned int reverse_mask = pow(2, (2 * K));
        reverse_mask = reverse_mask - 1;
        long long unsigned int one = 1;
        long long unsigned int check_mask = pow(2, (2 * K -1)); //check if the top bit is 1
        
    
        int total_valid_reads = 0;
        unsigned long long int total_nucleotides = 0;
        unsigned long long int total_kmers = 0;

        const int max_reads = 5000000;
		
		const int shift  = log2(M) - log2(BANKS/H);

        //IMPORTANT: ALL MEMORY GOING TO/FROM BOARD NEEDS TO BE 64 BIT ALIGNED
        //sequence* all_reads = (sequence*)malloc(sizeof(sequence) * max_reads);
		sequence* all_reads = (sequence*)aligned_alloc(64, sizeof(sequence) * max_reads);
		
		//sequence* all_reads_dev = (sequence*)malloc(sizeof(sequence) * max_reads);
        
        //query* all_qs = (query*)malloc(sizeof(query) * max_reads);
		query* all_qs = (query*)aligned_alloc(64, sizeof(query) * max_reads);
		memset(all_qs, 0, sizeof(query) * max_reads);
		printf("\nCombined size of the sequence (%lu bytes) and query (%lu bytes) buffers: %lu bytes\n", (sizeof(sequence) * max_reads), (sizeof(query) * max_reads), (sizeof(query) * max_reads) + (sizeof(sequence) * max_reads));
        
        short* kmer_len = (short*)malloc(sizeof(short) * max_reads);
        
        printf("\nStarting to parse now\n");
        //read the sequecing reads from the file
         while (ReadFASTA(ffp, &seq, &name, &L) && total_valid_reads < max_reads) {
			//printf("%d\n", L);

            if (L >= K){
                
                if(L > MAX_SEQ_L){
                    L = MAX_SEQ_L;
                }

                std::bitset<2*MAX_SEQ_L> seq_bitset = 0;
                std::bitset<2*MAX_SEQ_L> char_as_bits = 0;

                for (int i = 0; i < L; i++){
                    char_as_bits = 0; //any unknown char gets input as an A, should figure out what to actually do
                    if (seq[i] == 'a' || seq[i] == 'A'){
                        char_as_bits = 0;
                    }
                    else if(seq[i] == 'c' || seq[i] == 'C'){
                        char_as_bits = 1;
                    }
                    else if(seq[i] == 'g' || seq[i] == 'G'){
                        char_as_bits = 2;
                    }
                    else if(seq[i] == 't' || seq[i] == 'T'){
                        char_as_bits = 3;
                    }
                    seq_bitset <<= 2;
                    seq_bitset |= char_as_bits;
                }
                
                all_reads[total_valid_reads].seq = seq_bitset;
                all_reads[total_valid_reads].L = L;
                all_reads[total_valid_reads].valid = true;
				
                kmer_len[total_valid_reads] = (L- K + 1);
                
                total_valid_reads++;
                total_nucleotides += L;
                total_kmers += (L- K + 1);
            }
            free(seq);
            free(name);
        }
        
        CloseFASTA(ffp);

        
        // Create a queue bound to the chosen device.
        // If the device is unavailable, a SYCL runtime exception is thrown.
        printf("Going to attempt to create the queue\n");
		cl::sycl::property_list propList{cl::sycl::property::queue::enable_profiling()};
        sycl::queue q(device_selector, dpc_common::exception_handler, propList);
        printf("Queue has been made\n");

        // Print out the device information.
        std::cout << "Running on device: "
                  << q.get_device().get_info<info::device::name>() << "\n";
        
        
    
        //volatile char *bf = (char *)malloc_device(M/4, q);
        
        /*q.submit([&](handler& h) {
             h.memset((void *)bf, 0, M/4);

         });*/
        
        //buffer<char, 1> bf_buf{M/4}; 
		

        bool compose_finished = false;
		
		sycl::range<1> bytes_per_bank{(size_t)((M/(BANKS/H))/4)};
		
		std::vector<sycl::buffer<char,1>*> bank_bufs;
		
		printf("\nEach bank is getting %f MB\n", ((M/(BANKS/H))/4) * 0.000001);
		
		/*char* bank_arr[BF_BANKS];
		for (int i = 0; i < BF_BANKS; i++){
			bank_arr[i] = (char*)aligned_alloc(64,(M/(BANKS/H))/4);
			memset(bank_arr[i], 0, (M/(BANKS/H))/4);
		}*/
		
		//char* blank_bank = (char*)aligned_alloc(64,(M/(BANKS/H))/4);
		//memset(blank_bank, 0, (M/(BANKS/H))/4);
		//sycl::buffer<char,1> *blank_buf = new sycl::buffer<char,1>(blank_bank, bytes_per_bank);
		
		//for(int i = 0; i < BF_BANKS; i++){ 
		fpga_tools::UnrolledLoop<BF_BANKS>([&](auto i) {
            //sycl::buffer<char,1> *bank_buffer = new sycl::buffer<char,1>(bytes_per_bank);
			//sycl::buffer<char,1> *bank_buffer = new sycl::buffer<char,1>(bank_arr[i], bytes_per_bank,  ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<i+1>});
			__declspec( align(64) ) sycl::buffer<char,1> *bank_buffer = new sycl::buffer<char,1>(bytes_per_bank,  ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<i+1>});
			//bank_buffer[0] = { 0 };
            bank_bufs.push_back(bank_buffer);
        });
		
		
		
		//std::vector<char> blank_bank((M/(BANKS/H))/4, 0);
		
		//set all the banks to 0 intially
		fpga_tools::UnrolledLoop<BF_BANKS>([&](auto i) {
			q.submit([&](handler &h) {
				accessor bf_accessor(bank_bufs[i][0], h, write_only, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<i+1>});
				h.single_task([=]() {
					std::fill(bf_accessor.get_pointer(),bf_accessor.get_pointer() + ((M/(BANKS/H))/4),0);
					/*for (int i =0; i < ((M/(BANKS/H))/4); i++){
						bf_accessor[i] = 0;
					}*/
				});
			});
		});
		
		q.wait();
		
		
		
        
        

		buffer<sequence, 1> seq_buf{(all_reads), range{(unsigned long long)total_valid_reads}, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<0>}};
		
		auto start_sw = std::chrono::high_resolution_clock::now();

        auto e = q.submit([&](handler &h) {
            accessor input_accessor(seq_buf, h, read_only, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<0>});
            //size_t num_elements = input_buffer.get_count();

            h.single_task<produce_reads1>([=]() {
				//#pragma disable_loop_pipelining
				//for (int loop = 0; loop < 2; loop++){ //one loop for composition, next for query
					for (int j = 0; j < total_valid_reads; j++){
						//bool success = false;
						//do{
							//if(input_accessor[j].valid){
						sequence curr_seq = input_accessor[j];
						seq_pipe::write(curr_seq);
						/*if (loop == 1){ //now done with the reads, prepare to record the query data
							curr_seq.seq.set();
							curr_seq.L = 0;
							input_accessor[j] = curr_seq;
						}*/
							//}
						//}while(!success);
					}
				//}
            });
          });

                
        //split reads into cannonical k-mers and send to hash
        auto e2 = q.submit([&](handler& h) {
            h.single_task<parse_reads1>([=]()[[intel::kernel_args_restrict]]{
				//#pragma disable_loop_pipelining
				//for (int loop = 0; loop < 2; loop++){ //one loop for composition, next for query
				int loop = 1;
					for (int j = 0; j < total_valid_reads; j++){

						sequence dev_seq = seq_pipe::read();
						
						if(loop == 1){
                            fpga_tools::UnrolledLoop<H>([&](auto i) {
                                len2comp_pipes::PipeAt<i>::write(dev_seq.L);
                            });
                        }
						
						for(int i = 0; i < dev_seq.L - K + 1; i++){
							std::bitset<2*K> kmer_bits = 0;
							#pragma unroll
							for(int n = 0; n < K; n++){
								kmer_bits[n << 1] = dev_seq.seq[(i + n) << 1];
								kmer_bits[(n << 1) + 1] = dev_seq.seq[((i + n) << 1)+1];
							}
							if (kmer_bits[(K << 1) -1]){
								kmer_bits.flip();
							}

							std::bitset<MAX_K_BYTES * 8> kmer_bytes = 0;
							 #pragma unroll
							for (int s = 0; s < 2*K; s++){
								kmer_bytes[s] = kmer_bits[s];
							}
							
							kmer_data current_kmer;
                            current_kmer.kmer = kmer_bytes;
                            current_kmer.seq_num = j;
                            current_kmer.kmer_num = i;
							
							fpga_tools::UnrolledLoop<H>([&](auto hash_num) {
							
								kmer2hash_pipes::PipeAt<hash_num>::write(current_kmer);
								
							});
						}
					}
				//}
            });
        });
        
		
		//split the hashing off into it's own kernel
		fpga_tools::UnrolledLoop<1>([&](auto i) {
			q.submit([&](handler& h) {
				h.single_task([=]()[[intel::kernel_args_restrict]]{
					//out << "This is hashing kernel" << sycl::endl;
					long long unsigned int top_bits = (M >> 1) + (M >> 2) + (M >> 3);
					//#pragma disable_loop_pipelining
					//for (int loop = 0; loop < 2; loop++){
						int loop = 1;
						for (int j = 0; j < total_kmers; j++){
							kmer_data current_kmer = kmer2hash_pipes::PipeAt<i>::read();
							//for (int i =0; i < H; i++){
							//unsigned int result = MurmurHash2(&current_kmer.kmer, i) & (unsigned long long int)(M-1);
							unsigned long long int result = MurmurHash64A(&current_kmer.kmer, i) ;
							
                            result &= (unsigned long long int)(M-1);
							short bank_num = (result & top_bits) >> shift;
                            hash_result  tiny_result;
							tiny_result.result = (result & ((M-1) >> 3)); //this gets sent to the bank, strip top 2 bits
							//tiny_result.result = result;
							tiny_result.done = false;
							tiny_result.seq_num = current_kmer.seq_num;
							tiny_result.kmer_num = current_kmer.kmer_num;
							tiny_result.hash_num = i;
							
							//hash2bank_pipes::PipeAt<i>::write(tiny_result);
							
							fpga_tools::UnrolledLoop<8>([&](auto k) { //hate that this is hard coded
								if(bank_num == k){
									hash2bank_pipes::PipeAt<i * (BANKS/H) + k>::write(tiny_result);
								}
							});
							
							
						}
						
					
						hash_result  done_signal;
						done_signal.done = true;
						fpga_tools::UnrolledLoop<8>([&](auto k) { //hate that this is hard coded
							hash2bank_pipes::PipeAt<i*(BANKS/H) + k>::write(done_signal);
						});
					//}
				});
			});
		});
         
         //special case for the last one
        q.submit([&](handler& h) {
			h.single_task([=]()[[intel::kernel_args_restrict]]{
				//out << "This is hashing kernel" << sycl::endl;
				long long unsigned int top_bits = (M >> 1) + (M >> 2) + (M >> 3);
				//#pragma disable_loop_pipelining
				//for (int loop = 0; loop < 2; loop++){
					int loop = 1;
					for (int j = 0; j < total_kmers; j++){
						kmer_data current_kmer = kmer2hash_pipes::PipeAt<1>::read();
						//for (int i =0; i < H; i++){
						//unsigned int result = MurmurHash2(&current_kmer.kmer, i) & (unsigned long long int)(M-1);
						unsigned long long int result = MurmurHash64A(&current_kmer.kmer, 1) ;

						//result &= (unsigned long long int)(M-1);
						result = result % top_bits; //special case that lets us reuse the top_bits variable
						short bank_num = (result & top_bits) >> shift;
						hash_result  tiny_result;
						tiny_result.result = (result & ((M-1) >> 3)); //this gets sent to the bank, strip top 3 bits
						//tiny_result.result = result;
						tiny_result.done = false;
						tiny_result.seq_num = current_kmer.seq_num;
						tiny_result.kmer_num = current_kmer.kmer_num;
						tiny_result.hash_num = 1;

						//hash2bank_pipes::PipeAt<i>::write(tiny_result);

						fpga_tools::UnrolledLoop<7>([&](auto k) { //hate that this is hard coded
							if(bank_num == k){
								hash2bank_pipes::PipeAt<1 * (BANKS/H) + k>::write(tiny_result);
							}
						});


					}


					hash_result  done_signal;
					done_signal.done = true;
					fpga_tools::UnrolledLoop<7>([&](auto k) { //hate that this is hard coded
						hash2bank_pipes::PipeAt<1*(BANKS/H) + k>::write(done_signal);
					});
				//}
			});
		});
        
		fpga_tools::UnrolledLoop<BF_BANKS>([&](auto i) {
			auto acc_bf = q.submit([&](handler& h) {
				accessor bf_accessor(bank_bufs[i][0], h, read_write, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<i+1>});
				//accessor a_found(found_buf, h, write_only, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<2>, no_init});
				h.single_task([=]()[[intel::kernel_args_restrict]]{
					bool done = false;
					bool success;
					//#pragma disable_loop_pipelining
					//for (int loop = 0; loop < 2; loop++){ //one loop for composition, next for query
					int loop = 1;
						done = false;
						//[[intel::ivdep((bf_accessor))]]
						[[intel::ivdep]] [[intel::speculated_iterations(8)]]
						while(!done){
							//unsigned int result = res_pipe::read();
							hash_result read_result = hash2bank_pipes::PipeAt<i>::read(success);
							//hash_result read_result = hash2bank_pipes::PipeAt<i>::read();
							if (success){
								done = read_result.done;
								if(!done){
									short lower = (read_result.result & 3) << 1;
									unsigned int upper = read_result.result >> 2;
									char bf_byte = bf_accessor[upper];  //should split up this read and the write
									unsigned int idx = 1 << lower;

									if(loop == 0){ //composition stage

										if((bf_byte & idx) != 0){
											if((bf_byte & (idx << 1)) == 0){ //only write if we haven't already
												bf_byte = bf_byte | (idx << 1);
												bf_accessor[upper] |= bf_byte; //changed this, might be issue with atomic op
												//bf_accessor[upper] = bf_byte;
											}
										}
										else{ //if index is empty then write one to it
											bf_byte = bf_byte | idx;
											bf_accessor[upper] |= bf_byte; //changed for same reason
											//bf_accessor[upper] = bf_byte;
										}
									}
									else{ //the query stage

										bank_query q;


										q.found = (bf_byte & (idx << 1)) != 0;
										q.seq_num = read_result.seq_num;
										q.kmer_num = read_result.kmer_num;

										bank2comp_pipes::PipeAt<i>::write(q);

										//a_found[read_result.seq_num].found[read_result.kmer_num] = a_found[read_result.seq_num].found[read_result.kmer_num] & ((bf_accessor[upper] & (idx << 1)) != 0);
										//a_found[read_result.seq_num].seen++;

									}
								}
							}
						}
					//}
				});
			});
		});
		
		//need a kernel here to collect the query data from the banks
        //Collect the queries for a specific hash function, then pass that on to the global compilation
        fpga_tools::UnrolledLoop<1>([&](auto i) {
            q.submit([&](handler& h) {
                h.single_task([=]()[[intel::kernel_args_restrict]]{
                    
                    //out << "This is the compiler kernel" << sycl::endl;
                    bool early[8] = { 0 }; // array to check if one banks is ahead of the others and needs to wait
                    bank_query early_qs[8];
                    bank_query current_q;
					//bool problem = false; //here for debugging
                    for (int j = 0; j < total_valid_reads; j++){
                        int L = len2comp_pipes::PipeAt<i>::read();
                        std::bitset<MAX_SEQ_L - K + 1> found = 0;
						//std::bitset<2*MAX_SEQ_L> found = 0;
                        
                        for (int k = 0; k < L - K + 1; k++){
                            bool success = false;
                            int index = k & 7; //start at a new bank each time
							#pragma disable_loop_pipelining
                            while(!success){//check pipes until there one is read, need to make sure it belongs to the correct sequence
                                
                                fpga_tools::UnrolledLoop<8>([&](auto n) {

                                    if (index == n && !success) {
                                    //if (!success) {
                                        if (early[n]){//Shouldn't read from this pipe, already read one early
                                            if(early_qs[n].seq_num == j){ //no longer early
                                                current_q = early_qs[n];
                                                early[n] = false;
                                                success = true;
                                                found[current_q.kmer_num] = current_q.found; //add it to the query results
                                            }
                                            //else{
                                            //    index++;
                                            //}

                                        }
                                        else{ //if not early then its safe to try and read from that pipe
                                            current_q = bank2comp_pipes::PipeAt<i*(BANKS/H) + n>::read(success);
                                            if(success){//read result from pipe
                                                //success = true; //dummy line just to be safe
                                                if (current_q.seq_num != j){ // oops its early
                                                    early_qs[n] = current_q;
                                                    early[n] = true;
                                                    success = false;
                                                }
                                                else{ //add result to sequence bitset
                                                    found[current_q.kmer_num] = current_q.found;
                                                }
                                            }
                                            //else{ // no read :/
                                                //index++;
                                            //}
                                        }
										index++;
                                    }

                                    //index++;
                                });
								
								/*int sum = std::accumulate(early, early + 8, 0);
								if (sum == 8){ //all pipes are early, this is an issue
									problem = true;
									k = L; //end the current sequence
									success = true;
								}*/
                                
                                index = 0;
                                
                            }
                        }
                        
                        //finished reading all query results for this sequence, write these partial results to compiler
                        partial_query q;
                        q.found = found;
                        q.seq_num = j;
						//q.problem = problem;
                        comp2final_pipes::PipeAt<i>::write(q);
						//problem = false;
                    }
                        
                });
            });
        });
         //special case for the final hash function
        q.submit([&](handler& h) {
			h.single_task([=]()[[intel::kernel_args_restrict]]{

				//out << "This is the compiler kernel" << sycl::endl;
				bool early[7] = { 0 }; // array to check if one banks is ahead of the others and needs to wait
				bank_query early_qs[7];
				bank_query current_q;
				//bool problem = false; //here for debugging
				for (int j = 0; j < total_valid_reads; j++){
					int L = len2comp_pipes::PipeAt<1>::read();
					std::bitset<MAX_SEQ_L - K + 1> found = 0;
					//std::bitset<2*MAX_SEQ_L> found = 0;

					for (int k = 0; k < L - K + 1; k++){
						bool success = false;
						int index = k % 7; //start at a new bank each time
						#pragma disable_loop_pipelining
						while(!success){//check pipes until there one is read, need to make sure it belongs to the correct sequence

							fpga_tools::UnrolledLoop<7>([&](auto n) {

								if (index == n && !success) {
								//if (!success) {
									if (early[n]){//Shouldn't read from this pipe, already read one early
										if(early_qs[n].seq_num == j){ //no longer early
											current_q = early_qs[n];
											early[n] = false;
											success = true;
											found[current_q.kmer_num] = current_q.found; //add it to the query results
										}
										//else{
										//    index++;
										//}

									}
									else{ //if not early then its safe to try and read from that pipe
										current_q = bank2comp_pipes::PipeAt<1*(BANKS/H) + n>::read(success);
										if(success){//read result from pipe
											//success = true; //dummy line just to be safe
											if (current_q.seq_num != j){ // oops its early
												early_qs[n] = current_q;
												early[n] = true;
												success = false;
											}
											else{ //add result to sequence bitset
												found[current_q.kmer_num] = current_q.found;
											}
										}
										//else{ // no read :/
											//index++;
										//}
									}
									index++;
								}

								//index++;
							});
							
							/*int sum = std::accumulate(early, early + 7, 0);
							if (sum == 7){ //all pipes are early, this is an issue
								problem = true;
								k = L; //end the current sequence
								success = true;
							}*/
							
							index = 0;

						}
					}

					//finished reading all query results for this sequence, write these partial results to compiler
					partial_query q;
					q.found = found;
					q.seq_num = j;
					//q.problem = problem;
					comp2final_pipes::PipeAt<1>::write(q);
					//problem = false;
				}

			});
		});
        
		

		
		{ //Get results from all banks and combine to form the final query results
        buffer<query, 1> found_buf{all_qs, range{(unsigned long long)total_valid_reads}, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<0>}}; 
        auto final_comp = q.submit([&](handler& h) {
            //accessor a_found{found_buf, h, read_write, no_init};
			accessor a_found{found_buf, h, read_write};
			//accessor a_found(seq_buf, h, read_write, ext::oneapi::accessor_property_list{sycl::ext::oneapi::no_alias, ext::intel::buffer_location<0>});
            h.single_task([=]()[[intel::kernel_args_restrict]]{
                partial_query curr_results;
				query full_query;
                for (int j = 0; j < total_valid_reads * H; j++){ //might be an issue here, double check this line
                    bool success = false;
                    int index = j & (H - 1);
					#pragma disable_loop_pipelining
                    while(!success){
                        fpga_tools::UnrolledLoop<H>([&](auto i) {
                            if (index == i && !success){
                                curr_results = comp2final_pipes::PipeAt<i>::read(success);
								index++;
                            }
                            //index++;
                        });
                        index = 0;
                    }
					full_query = a_found[curr_results.seq_num];
					if(full_query.seen == 0){
						full_query.seq = curr_results.found;
					}
					else{
						full_query.seq = full_query.seq & curr_results.found;
					}
                    full_query.seen++;
					//full_query.problem |= curr_results.problem;
                    //a_found[curr_results.seq_num].seen++;
					a_found[curr_results.seq_num] = full_query;
                } 
            });
        });
		//}
        
		
        
        q.wait();
		
		auto end_sw = std::chrono::high_resolution_clock::now();
		

        // Put Through Database
        compose_finished = true;
        std::unordered_map<std::bitset<2*K>, int> *hash_table = new std::unordered_map<std::bitset<2*K>, int>;
        
        sycl::host_accessor a_found{found_buf};

        
        for (int j = 0; j < total_valid_reads; j++){
            //while( H != all_qs[j].seen){};
			while( a_found[j].seen < 0){}; //currently hangs on this, maybe L isn't being set to 0?
			/*if (a_found[j].seen != H){
				printf("Oops, potential error on queries for read # %d  where seen = %d\n", j, a_found[j].seen);
			}
			if (j % 10000 == 0){
				std::cout << "Query #" << j << ": " << a_found[j].seq << std::endl;
			}*/
			/*if (all_reads_dev[j].problem){
				printf("Problem flag has been raised for read # %d \n", j);
			}*/
            for(int i = 0; i <  all_reads[j].L - K + 1; i++){
                if(a_found[j].seq[i] == 1){
                    std::bitset<2*K> kmer_bits = 0;
                    for(int n = 0; n < K; n++){
                        kmer_bits[n << 1] = all_reads[j].seq[(i + n) << 1];
                        kmer_bits[(n << 1) + 1] = all_reads[j].seq[((i + n) << 1)+1];
                    }
                    if (kmer_bits[(K << 1) -1]){
                        kmer_bits.flip();
                    }
                    //unsigned long long int kmer_as_int = kmer_bits.to_ullong();
                    std::unordered_map<std::bitset<2*K>, int>::iterator it = hash_table->find(kmer_bits);
                    if (it != hash_table->end()) {
                            it->second++;
                    }
                    else{
                        hash_table->insert({kmer_bits, 1});
                    }
                }
            }
        }
        


        long long unsigned int distinct_hashed = (long long unsigned int)hash_table->size();

        
        //remove false positives
        std::unordered_map<std::bitset<2*K>, int>::iterator itr; 
        for(itr = hash_table->begin(); itr !=  hash_table->end();){
            if (itr->second == 1){
                itr = hash_table->erase(itr);
            }
            else{
                ++itr;
            }
        }
        
        
        //calc the false positive rate
        
        double fp = distinct_hashed - hash_table->size();
        

        double kmers_in_table = 0;
        std::unordered_map<std::bitset<2*K>, int>::iterator itr2; 
        for(itr2 = hash_table->begin(); itr2 !=  hash_table->end();){
            kmers_in_table += itr2->second;
            itr2++;
        }
        double fp_rate = fp / (total_kmers - kmers_in_table);
        fp_rate = fp_rate * 100; 
        
        printf("\nNumber of distinct hashed k_mers: %llu", distinct_hashed);
        printf("\nNumber remaining after removing fp: %llu", (long long unsigned int)hash_table->size());
        printf("\nFalse positive rate: %f %%", fp_rate);
        printf("\nNumber of valid reads: %d", total_valid_reads);
		auto dur_sw = std::chrono::duration_cast<std::chrono::nanoseconds>(end_sw - start_sw);
		double exe_time = dur_sw.count() / 1000000000.0;
		printf("\nNumber of %d-mers: %llu", K, total_kmers);
		printf("\nKernel time : %f s", exe_time);
		printf("\nMillions of %d-mers per second : %f\n", K, (total_kmers / exe_time) / 1000000);
		
		uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
		uint64_t end = final_comp.get_profiling_info<sycl::info::event_profiling::command_end>();
		double duration = static_cast<double>(end - start) / NSEC_IN_SEC;
		std::cout << "Kernel execution time w/ new method: " << duration << " sec" << std::endl;
		printf("Millions of %d-mers per second w/ new method : %f\n", K, (total_kmers / duration) / 1000000);
		
        //printf("\nNumber of nucleotides: %llu", total_nucleotides);
        //print_hash_lookup_table(hash_table);
        // Compute total time of execution
		
        
        //std::cout << "\nExecution Time: " << dur_sw.count() / 1000000000.0 << " s" << std::endl;

        delete hash_table;
		
		}
        
        
        
        //free((void *)bf, q);
        
		//free(all_reads_dev);
        
        
       
        printf("\n");
		
		free(all_reads);
		free(kmer_len);
		free(all_qs);
        return 0;
    }
    catch (sycl::exception const& e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

}


