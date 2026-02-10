#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int lzcomplexity(char *ss){
  int ii = 0, kk = 1, el = 1, kmax = 1, cc = 1, nn;
  nn = strlen(ss);
  while (1){
    if (ss[ii + kk - 1] == ss[el + kk - 1]){
      kk++;
      if ((el + kk) > nn){
        ++cc;
        break;
      }
    } else {
      if ( kk > kmax ){
        kmax = kk;
      }
      ++ii;
      if (ii == el){
        ++cc;
        el += kmax;
        if ((el + 1) > nn){
          break;
        }
        ii = 0;
        kk = 1;
        kmax = 1;
      } else {
        kk = 1;
      }
    }
  }
  return cc;
}

int compute_lzc_from_events(const int *events, int num_events)
{
    /* allocate string: one char per event + null terminator */
    char *spike_seq_string = (char *)malloc(num_events + 1);
    if (!spike_seq_string) {
        return -1;  /* allocation failure */
    }

    /* equivalent to: ''.join(map(str, spike_seq.tolist())) */
    for (int i = 0; i < num_events; i++) {
        spike_seq_string[i] = events[i] ? '1' : '0';
    }
    spike_seq_string[num_events] = '\0';

    /* equivalent to: lempel_ziv_complexity(spike_seq_string) */
    int lz_score = lzcomplexity(spike_seq_string);

    free(spike_seq_string);
    return lz_score;
}

int test(){
  char data[256] = "1001111011000010\0";
  printf("lzc=%d\n", lzcomplexity(data));
  return 0;
}

int main(int argc, char **argv){
  FILE * input;
  char * buffer;
  int size;
  int length;
  if (argc < 2){
    printf("Require an input file. Usage %s filename\n", argv[0]);
    return -1;
  } 
  input = fopen(argv[1], "rb");
  fseek(input, 0L, SEEK_END);
  size = ftell(input);
  printf("File size: %d\n", size);
  fclose(input);
  input = fopen(argv[1], "r");
  buffer = (char*) calloc(size+1, sizeof(char));
  fgets(buffer, size, input);  
  fclose(input);
  printf("Read %s\n", buffer);  
  length = strlen(buffer);
  printf("Length of string including terminal new line: %d\n", length);
  if (buffer[length - 1] == '\n'){
    buffer[length - 1] = '\0';
    length--;
  }
  printf("Length of string without terminal new line: %d\n", length);
  printf("Lempel-Ziv complexity of this string of length %d is %d\n", length, lzcomplexity(buffer));    
    return 0;
}
            
        


/* lzcomplexity.c ends here */