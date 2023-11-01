# EDS

EDS: An Effective DNA File Storage System to Practically Archive and Retrieve Medical Data

## What is EDS?

EDS proposes an **e**ffective **DNA** **s**torage (EDS) approach for archiving medical data. The EDS approach incorporates (i) a novel fraction strategy for handling the crucial problem of rotating encoding to control data loss and DNA sequencing costs; (ii) a novel rule-based quaternary transcoding to satisfy bio-constraints and ensure reliable mapping; and (iii) a new indexing method to simplify random search and access. The approach's effectiveness is validated through computer simulation and biological experiments, confirming its practicality. 


# Installation 

Step-by-step installation is as follows: 

## Tools and environment 

> Install Python IDE, PyCharm from here https://www.jetbrains.com/pycharm/download/?section=windows,

> Install following Python packages

- Codecs
- Math
- Struct
- Os, random
- Binascii
- Blast
- Struct


## Experimental steps 

Update the existing system according to requirements or run.

### ENCODING 
1.	Open EDS.py
2.	The default settings are for encoding the image files (16 chunks of MRI). Users can change the input file path and output results path at img_dir = './image/' and result_dir = './imageResults/', respectively. 
3.	If the user wants to encode a non-image file, the first user has to turn the '__main__' function on by removing # from line 633 and turning off the function by inserting # in front of line 632. 
4.	Suppose the user runs the code for image file encoding; the following output can be found in the terminal;

   - Original binary segment 
   - Max GC 
   - Min GC 
   - Total GC
   - Max length 
   - Min length 
   - Average length 
   - Total sequences
   - Density 
   - Time
   - Maximum file size
   - Adding sequences from FASTA; added x sequences in x seconds.

5.	The folder 'imageResults' has 16 subfolders of 16 corresponding chunks. Each sub-image has primers and DNA sequences generated by the code. For experimental convince, we have merged all the chunks images in 'result.dna' file in the 'imageResults' folder. (merged sequences can be differentiated by the primer difference) The result.dna was converted into an xlsx file to send out for gene synthesis. 
6.	Suppose the user runs the code for non-image file encoding (i.e., report); after setting the pdf_dir path, the code will provide the results.dna file in the 'reportResults' folder.  


The resulting xlsx files were sent out to DNA synthesis companies. The synthesized DNA and gene were sequenced from another company, and later, we received the DNA sequences with multiple results. These DNA sequences were decoded to access the required chunks and different files.  

### DECODING

1.	Open decode_one.py
2.	Select the '__main__' function for image and non-image files on lines 226 and 227. 
3.	Provide the input_path of a file which is being decoded. 
4.	The decoded results will be generated back to the original folders.


In the manuscript, we have offered various analyses on DNA and binary file recovery, running time, memory utilization, GC and RC constraints satisfactions, and biological validation, for which readers are referred to the main draft and supplementary file.



# License

EDS is licensed under the GNU General Public License; for more information, read the LICENSE file or refer to:

http://www.gnu.org/licenses/

# Citation

A related paper is submitted to the SCI journal. 
