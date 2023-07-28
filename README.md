# EDS

An Effective DNA File Storage System to Practically Archive Medical Data

## What is EDS?

EDS proposes an **e**ffective **DNA** **s**torage (EDS) approach for archiving medical data. The EDS approach incorporates (i) a novel fraction strategy for handling the crucial problem of rotating encoding to control data loss and DNA sequencing costs; (ii) a novel rule-based quaternary transcoding to satisfy bio-constraints and ensure reliable mapping; and (iii) a new indexing method to simplify random search and access. The approach's effectiveness is validated through computer simulation and biological experiments, confirming its practicality. 


# Installation 

Step-by-step installation is as follows: 

## Tools and environment 

> Install Python IDE, PyCharm from here https://www.jetbrains.com/pycharm/download/?section=windows,

> Install Python following packages

- Codecs
- Math
- Struct
- Os, random
- Binascii
- Blast
- Struct


## Experimental steps 

Update the existing system according to requirements or run.
The EDS.py code, as provided, performs image file transcoding by dividing it into 16 equal segments. Nevertheless, users have the flexibility to adapt the code for different file types, as EDS accommodates various formats. Upon execution, the user gains access to the following information: 
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
Moreover, the output DNA sequences in .fa and .blast formats are accessible in the "imageResults" directory (for image files) and the "reportResults" directory (for non-image files). These DNA sequences serve as essential components for the subsequent DNA synthesis and decoding procedures.


# License

BO-DNA is licensed under the GNU General Public License; for more information, read the LICENSE file or refer to:

http://www.gnu.org/licenses/

# Citation

A related paper is under review. 
