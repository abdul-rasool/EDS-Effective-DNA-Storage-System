import codecs
import math
import struct
import os, random
import binascii

# return GC ratio
def check_gc(sequence):
    return float(sequence.count("C") + sequence.count("G")) / float(len(sequence))

# Binary segmentation
def load_file_to_BinaryMatrix(path, segment_length):
    with open(path, mode="rb") as file:
        size = os.path.getsize(path)
        matrix = [['0' for _ in range(segment_length)] for _ in range(math.ceil(size * 8 / segment_length))]
        row = 0
        col = 0
        for byte_index in range(size):
            # Read a file as bytes
            one_byte = file.read(1)
            element = list(str(bin(struct.unpack("B", one_byte)[0]))[2:].zfill(8))
            for bit_index in range(8):
                matrix[row][col] = element[bit_index]
                col += 1
                if col == segment_length:
                    col = 0
                    row += 1
        for i in range(row+1):
            matrix[i] = "".join(matrix[i])
    return matrix, size
def read_by_str(path, segment_length):
    f = codecs.open(path, 'rb')
    size = os.path.getsize(path)
    binary_list = []
    with f:
        fileText = f.read()
        hexstr = binascii.hexlify(fileText)
        bsstr = bin(int(hexstr, 16))[2:]
        str_binary = bsstr
        binary_length = len(bsstr)
    pos = 0
    i = 0
    while pos < binary_length:
        binary_list.append(str_binary[pos:pos+segment_length])
        pos += segment_length
        i+=1
    last_len = len(binary_list[-1])
    if last_len != segment_length:
        while last_len !=segment_length:
            binary_list[-1] += '0'
            last_len +=1
    return binary_list,size

B_set = [ [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['2','5','0','3'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['2','5','3','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','2','5','3'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['2','0','5','3'],'GG':['3','0','2','5']}],
                [{'AA':['5','2','0','3'],'TT':['3','5','0','2'],'CC':['0','3','5','2'],'GG':['0','3','2','5']},{'AA':['5','2','3','0'],'TT':['3','5','2','0'],'CC':['3','0','5','2'],'GG':['3','0','2','5']}] ]
E_set = {'AA':{'0': 'TC', '1': 'TG', '00': 'CA', '01': 'CT', '10': 'GA', '11': 'GT'},
             'TT':{'0': 'AC', '1': 'AG', '00': 'CA', '01': 'CT', '10': 'GA', '11': 'GT'},
             'CC':{'0': 'AC', '1': 'AG', '00': 'TC', '01': 'TG', '10': 'GA', '11': 'GT'},
             'GG':{'0': 'AC', '1': 'AG', '00': 'TC', '01': 'TG', '10': 'CA', '11': 'CT'}}

# new Quaternary code
def n_Q_like(input_str, B_set):
    dna_str = ""
    rule = 'ATCG'
    A_extra = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
    B_extra = {'00': 'G', '01': 'C', '10': 'T', '11': 'A'}
    # first two bases
    dna_str += A_extra[input_str[0:2]] + A_extra[input_str[2:4]]
    p1 = 4
    p2 = 0
    flag1 = 1
    flag2 = 1
    str_len = len(input_str)
    while p1 < str_len:
        if p1 == str_len - 2:
            # dna_str += end[input_str[p1:]]
            dna_str += solve_end(input_str=dna_str,E_set=E_set, bin_str=input_str[p1:])
            break
        elif p1 == str_len - 1:
            # dna_str += end[input_str[p1]]
            dna_str += solve_end(input_str=dna_str,E_set=E_set, bin_str=input_str[p1])
            break
        else:
            pre_two = dna_str[p2] + dna_str[p2 + 1]
            if flag2 % 2 !=0:
                current_set = B_set[0]
            else:
                current_set = B_set[1]
            if pre_two in current_set.keys():
                if input_str[p1] == '0':
                    zero_index = current_set[pre_two].index('0')
                    dna_str += rule[zero_index]
                    p1 += 1
                    p2 += 1
                    flag1 = -flag1
                    flag2+=1
                else:
                    if p1 == str_len - 1:
                        # dna_str += end[input_str[p1]]
                        dna_str += solve_end(input_str=dna_str, E_set=E_set, bin_str=input_str[p1])
                    else:
                        for index, value in enumerate(current_set[pre_two]):
                            temp = 2 ** 1 * int(input_str[p1]) + 2 ** 0 * int(input_str[p1 + 1])
                            if value == str(temp):
                                dna_str += rule[index]
                                p2 += 1
                        p1 += 2
                        flag1 = -flag1
                        flag2 += 1
            else:
                if p1 != str_len - 1:
                    ss = input_str[p1:p1 + 2]
                    if flag1 == -1:
                        dna_str += B_extra[ss]
                    else:
                        dna_str += A_extra[ss]
                    p2 += 1
                    p1 += 2
                else:
                    # dna_str += end[input_str[p1]]
                    dna_str += solve_end(input_str=dna_str, E_set=E_set, bin_str=input_str[p1])
                    p1 += 1
    return dna_str

# last 1 or 2 bits convert rule
def solve_end(input_str, E_set, bin_str):
    bases = input_str[-2:]
    if bases in E_set.keys():
        end = E_set[bases]
    else:
        end = {'0': 'AC', '1': 'TC', '00': 'CA', '01': 'GT', '10': 'AG', '11': 'TG'}
    return end[bin_str]

# choose best rule id
def best_rule(input_str,rule_set):
    record = []
    for i in range(16):
        set = rule_set[i]
        result = n_Q_like(input_str, set)
        result_len = len(result)
        gc_ratio = float(result.count("C") + result.count("G")) / float(result_len)
        info_dict = dict.fromkeys(['result_len', 'gc_ratio', 'dna_sequence'])
        info_dict['result_len'] = result_len
        info_dict['gc_ratio'] = gc_ratio
        info_dict['dna_sequence'] = result
        record.append(info_dict)
    candicate_id = 0
    min_bases = 200
    for j in range(16):
        if record[j]['gc_ratio'] >= 0.4 and record[j]['gc_ratio'] <= 0.6:
            if record[j]['result_len'] <= min_bases:
                candicate_id = j
                min_bases = record[j]['result_len']
    return candicate_id, record[candicate_id]['dna_sequence']

# decimal to Ternary
def dec_to_ter(num):
    l = []
    if num < 0:
        return "- " + dec_to_ter(abs(num))
    else:
        while True:
            num, reminder = divmod(num, 3)
            l.append(reminder)
            if num == 0:
                return l[::-1]

# index encode(Ternary Goldman)
def index_encode(index, index_length):
    pre = 'A'
    index_result = ""
    base_rule = {'A':['C','G','T'],'T':['A','C','G'],'C':['G','T','A'],'G':['T','A','C']}
    ter_index = list(map(int, list("".join(list(map(str, dec_to_ter(index)))).zfill(index_length))))
    for num in ter_index:
        base = base_rule[pre][num]
        index_result += base
        pre = base
    return index_result

# 4 bases connect index and data
def rule_encode(rule):
    result = ""
    encode_rule = [{'0':'A','1':'T'},{'0':'C','1':'G'},{'0':'A','1':'T'},{'0':'C','1':'G'}]
    binary = str(bin(rule))[2:].zfill(4)
    for i in range(4):
        result += encode_rule[i][binary[i]]
    return result

# divide 16 parts
def part_index(part):
    encode_rule = {'00':'A','01':'T','10':'C','11':'G'}
    binary = str(bin(part))[2:].zfill(4)
    result = encode_rule[binary[:2]] + encode_rule[binary[2:]]
    return  result

def decode_part(dna_part):
    decode_rule = {'A':'00','T':'01','C':'10','G':'11'}
    dna_part = list(dna_part)
    binary = decode_rule[dna_part[0]] + decode_rule[dna_part[1]]
    return binary

# encode file dictory
def encode(binary_data,Rule_set,part_id):
    seg_num = len(binary_data)
    print('Original binary segment total',seg_num)
    sequence_record = []
    density_record = []
    low_record = []
    gc_record = []
    dna_record = []
    maxlen = 0
    minlen = 1000
    maxgc = 0.01
    mingc = 0.99
    total_gc = 0
    total_Density = 0
    index_length = len(dec_to_ter(seg_num))
    part_bases = part_index(part_id)
    for i in range(seg_num):
        best_id, best = best_rule(binary_data[i], Rule_set)
        # dna = n_Q_like(binary_data[i],Rule_set)
        if len(best) > maxlen:
            maxlen = len(best)
        if len(best) < minlen:
            minlen = len(best)
        sequence = part_bases + index_encode(i,index_length) +rule_encode(best_id) + best
        sequence_record.append(sequence)
        total_Density += len(best)+index_length+4
        density = 180/(len(best)+index_length+4)
        density_record.append(density)
        if density <1.6:
            low_record.append(binary_data[i])
        gc_ratio = check_gc(best)
        total_gc += gc_ratio
        if gc_ratio > maxgc:
            maxgc = gc_ratio
        if gc_ratio < mingc:
            mingc = gc_ratio
        if gc_ratio <0.4 or gc_ratio >0.6:
            gc_record.append(gc_ratio)
            dna_record.append(binary_data[i])
    print('Max GC',maxgc)
    print('Min GC',mingc)
    print('Max length', maxlen)
    print('Min length', minlen)
    print('total GC',total_gc/seg_num)
    print('density:', seg_num * 180/ total_Density)
    print('avg length',total_Density/seg_num)
    return density_record,low_record,gc_record,sequence_record

# encode file
def encode_simple(binary_data,Rule_set):
    seg_num = len(binary_data)
    print('Original binary segment total',seg_num)
    sequence_record = []
    density_record = []
    low_record = []
    gc_record = []
    dna_record = []
    maxlen = 0
    minlen = 1000
    maxgc = 0.01
    mingc = 0.99
    total_gc = 0
    total_Density = 0
    index_length = len(dec_to_ter(seg_num))
    for i in range(seg_num):
        best_id, best = best_rule(binary_data[i], Rule_set)
        # dna = n_Q_like(binary_data[i],Rule_set)
        if len(best) > maxlen:
            maxlen = len(best)
        if len(best) < minlen:
            minlen = len(best)
        sequence = index_encode(i,index_length) +rule_encode(best_id) + best
        sequence_record.append(sequence)
        total_Density += len(best)+index_length+4
        density = 180/(len(best)+index_length+4)
        density_record.append(density)
        if density <1.6:
            low_record.append(binary_data[i])
        gc_ratio = check_gc(best)
        total_gc += gc_ratio
        if gc_ratio > maxgc:
            maxgc = gc_ratio
        if gc_ratio < mingc:
            mingc = gc_ratio
        if gc_ratio <0.4 or gc_ratio >0.6:
            gc_record.append(gc_ratio)
            dna_record.append(binary_data[i])
    print('Max length', maxlen)
    print('Min length', minlen)
    print('total GC',total_gc/seg_num)
    print('density:', seg_num * 180/ total_Density)
    print('avg length',total_Density/seg_num)
    return density_record,low_record,gc_record,sequence_record

# write sequences to text file
def write_dna_file(path, dna_sequences):
    try:
        os.makedirs(os.path.dirname(path))
    except:
        pass

    with open(path, "w") as file:
        for index, dna_sequence in enumerate(dna_sequences):
            _out = ">seq{}\n{}\n".format(index, "".join(dna_sequence))
            file.write(_out)


# read DNA file
def read_dna_file(path):
    dna_sequences = []
    with open(path, "r") as file:
       #Read line by line
        lines = file.readlines()
        for  index, line in enumerate(lines):
            dna_sequences.append(list(line.replace("\n", "")))
    return dna_sequences

# decode rule ID
def rule_decode(bases):
    binary_str = ""
    decode_rule = [{'A':'0','T':'1'},{'C':'0','G':'1'},{'A':'0','T':'1'},{'C':'0','G':'1'}]
    for i in range(4):
        binary_str += decode_rule[i][bases[i]]
    rule = int(binary_str,2)
    return rule

# decode sequence
def decode_index(input_str, index_length):
    ter_index = ''
    pre = 'A'
    base_rule = {'A': ['C', 'G', 'T'], 'T': ['A', 'C', 'G'], 'C': ['G', 'T', 'A'], 'G': ['T', 'A', 'C']}
    pos = 0
    while pos < index_length:
        ter_index += str(base_rule[pre].index(input_str[pos]))
        pre = input_str[pos]
        pos += 1
    ter_list = list(map(int,list(ter_index)))
    index = 0
    for i in range(index_length):
        index += 3 ** (index_length - i - 1) * ter_list[i]
    return index

# decode Q-like (payload)
def q_like_decode(input_dna, B_set, rule_index, E_set):
    binary_str = ''
    rule = B_set[rule_index]
    base_to_bin_A = {'A':'00','T':'01','C':'10','G':'11'}
    base_to_bin_B = {'G':'00','C':'01','T':'10','A':'11'}
    base_rule = ['A','T','C','G']
    dec_to_bin = {'0':'0','2':'10','3':'11'}
    binary_str += base_to_bin_A[input_dna[0]] + base_to_bin_A[input_dna[1]]
    base_len = len(input_dna)
    flag1 = 1
    flag2 = 1
    pos = 2
    pre = input_dna[0]+input_dna[1]
    while pos != base_len-2:
        if pre not in rule[0].keys():
            if flag2 %2 == 1:
                binary_str += base_to_bin_A[input_dna[pos]]
            else:
                binary_str += base_to_bin_B[input_dna[pos]]
            pre = input_dna[pos-1]+input_dna[pos]
            pos += 1
        else:
            if flag1 == 1:
                binary_str += dec_to_bin[rule[0][pre][base_rule.index(input_dna[pos])]]
            else:
                binary_str += dec_to_bin[rule[1][pre][base_rule.index(input_dna[pos])]]
            pre = input_dna[pos-1]+input_dna[pos]
            flag1 = -flag1
            flag2 += 1
            pos += 1
    if pre in E_set.keys():
        end = E_set[pre]
    else:
        end = {'0': 'AC', '1': 'TC', '00': 'CA', '01': 'GT', '10': 'AG', '11': 'TG'}
    end_two = input_dna[-2]+input_dna[-1]
    for key,value in end.items():
        if end_two == value:
            binary_str += key
    return binary_str

# To get orignal file
def write_BinaryMatrix_to_file(path, matrix, size):
    with open(path, "wb+") as file:
        # Change bit to byte (8 -> 1), and write a file as bytes
        bit_index = 0
        temp_byte = 0
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                bit_index += 1
                temp_byte *= 2
                temp_byte += matrix[row][col]
                if bit_index == 8:
                    if size > 0:
                        file.write(struct.pack("B", int(temp_byte)))
                        bit_index = 0
                        temp_byte = 0
                        size -= 1
    return True

# Decode
def decode(dna_sequences, B_set, E_set):
    seg_num = len(dna_sequences)
    matrix = [[] for _ in range(seg_num)]
    index_length = len(dec_to_ter(seg_num))
    # each DNA segment
    for sequence in dna_sequences:
        # First: decode index
        index = decode_index(sequence[:index_length],index_length)
        # Second: decode rule
        rule_index = rule_decode(sequence[index_length:index_length+4])
        # Third: decode payload
        binary_data = q_like_decode(sequence[index_length+4:],B_set=B_set,rule_index = rule_index,E_set=E_set)
        matrix[index] = list(map(int,binary_data))
    return matrix

# Add XOR redundancy
def redundancy(data,sequence_len):
    with_redundancy = []
    last_comp = []
    data_len = len(data)
    # if sequence nums is odd, last one add complementary redundancy
    if data_len % 2 != 0:
        data_len -= 1
        last_comp.append(data[-1])
        c_data = []
        data_list = list(data[-1])
        for i in range(sequence_len):
            if data_list[i] == '0':
                c_data.append('1')
            else:
                c_data.append('0')
        last_comp.append("".join(c_data))
    for j in range(0,data_len-1,2):
        with_redundancy.append(data[j])
        with_redundancy.append(data[j+1])
        a_list = list(data[j])
        b_list = list(data[j+1])
        c_list = []
        for k in range(sequence_len):
            if a_list[k] == b_list[k]:
                c_list.append('0')
            else:
                c_list.append('1')
        with_redundancy.append("".join(c_list))
    if len(last_comp)!=0:
        with_redundancy += last_comp
    return with_redundancy

def decimal2OtherSystem(decimal_number: int, other_system:int, precision:int = None) -> list:
   

    remainder_list = []
    while True:
        remainder = decimal_number%other_system
        quotient = decimal_number//other_system
        remainder_list.append(str(remainder))
        if quotient == 0:
            break
        decimal_number = quotient


    num_list = remainder_list[::-1]
  #Specify precision
    if precision != None:
        if precision < len(num_list):
            raise ValueError("The precision is smaller than the length of number. Please check the [precision] value!")
        else:
            num_list = ["0"]*(precision - len(num_list)) + num_list
    return num_list

def getHomoLen(seq: str) -> int:
   
    seq_new = seq + "$"
    pos1 = 0
    pos2 = 1
    max_len = 0
    while pos1 < len(seq):
        while seq_new[pos2] == seq_new[pos1]:
            pos2 += 1
        max_len = max(max_len, pos2-pos1)
        pos1, pos2 = pos2, pos2+1

    return max_len

def getPrimerList(save_path, primer_len:int=20, primer_num:int = 100, homo:int = 3, gc=0.5):
    
    mapping_rule = {"0":"A", "1":"C", "2":"G", "3":"T"}

    primer_set = set()
    max_num = int('3'*primer_len, 4)

    f = open(save_path, "w")
    n = 0
    while True:
        primer_decimal_num = random.randint(0, max_num)
        primer_quaternary_num_list = decimal2OtherSystem(primer_decimal_num, 4, primer_len)
        primer_seq = "".join([mapping_rule[i] for i in primer_quaternary_num_list])

        # filter
        if (primer_seq.endswith("A") or primer_seq.endswith("T")):
            continue
        if abs((primer_seq.count("G")+primer_seq.count("C"))/primer_len-gc) > 0.1:
            continue
        if getHomoLen(primer_seq) > homo:
            continue

        primer_set.add(primer_seq)
        _out = ">seq_{}\n{}\n".format(n, primer_seq)
        f.write(_out)
        n += 1

        if len(primer_set) >= primer_num:
            break
    f.close()

    return list(primer_set)

def runBlast(ref_path, primer_path):
   
    result_path = os.path.dirname(ref_path) + os.sep + "result.blast"
    shell = "makeblastdb -dbtype nucl -in {}\n".format(ref_path)
    shell += "blastn -query {} -db {} -out {} -outfmt 6\n".format(primer_path, ref_path, result_path)
    # print(shell)
    os.system(shell)

    return result_path

def filterPrimer(blast_path, primer_list):
   

    homology_index = []
    with open(blast_path) as f:
        for i in f:
            _index = i.strip().split("\t")[0]
            _index = int(_index.split("_")[1])
            homology_index.append(_index)

    # chose primer
    primer_select_list = []
    for _index, primer in enumerate(primer_list):
        if _index in homology_index:
            continue
        primer_select_list.append(primer)

        if len(primer_select_list) == 2:
            break

    print("primer_F: {}\nprimer_R: {}\n".format(primer_select_list[0], primer_select_list[1]))
    return primer_select_list


def addPrimer(primer_f, primer_r, sequences):
    compement_dic = {"A":"T", "T":"A", "C":"G", "G":"C"}
    add_primer_r = "".join([compement_dic[i] for i in primer_r][::-1])
    with_primer_seq_list = ["{}{}{}".format(primer_f, i, add_primer_r) for i in sequences]
    return with_primer_seq_list
#for image encode
def main_for_img():
    img_dir = './image/'
    result_dir = './imageResults/'
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)

    file_name_list = os.listdir(img_dir)
    file_name_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    count = 0
    # 16 part sequences
    final_sequence_list = []
    binary_segment_length = 1040
    for filename in file_name_list:
        print(filename)
        file_path = img_dir + filename
        # read one file
        data, size = read_by_str(file_path, binary_segment_length)
        r_data = redundancy(data, binary_segment_length)
        density_r, low, gc, sequences = encode(r_data, B_set, count)
        # final_sequence += sequences
        count += 1
        _save_path = "{}/{}/result.fa".format(result_dir, count)
        write_dna_file(_save_path, sequences)

        primer_path = os.path.dirname(_save_path) + os.sep + "primers.fa"
        primer_list = getPrimerList(primer_path)
        blast_path = runBlast(_save_path, primer_path)
        primer_f, primer_r = filterPrimer(blast_path, primer_list)
        _with_primer_seq_list = addPrimer(primer_f, primer_r, sequences)
        final_sequence_list += _with_primer_seq_list

    save_path = "{}/result.dna".format(result_dir)
    write_dna_file(save_path, final_sequence_list)
# for non-image encode
def main_for_pdf():
    pdf_dir = 'MRI_report_2.pdf'
    result_dir = './reportResults/'
    binary_segment_length = 180
    data, size = load_file_to_BinaryMatrix(pdf_dir, binary_segment_length)
    # size should record
    print(size)
    r_data = redundancy(data, binary_segment_length)
    density_r, low, gc, sequences = encode_simple(r_data, B_set)
    save_path = "{}/result.fa".format(result_dir)
    write_dna_file(save_path, sequences)

    primer_path = os.path.dirname(save_path) + os.sep + "primers.fa"
    primer_list = getPrimerList(primer_path)
    blast_path = runBlast(save_path, primer_path)
    primer_f, primer_r = filterPrimer(blast_path, primer_list)
    with_primer_seq_list = addPrimer(primer_f, primer_r, sequences)
    primer_save_path = "{}/result.dna".format(result_dir)
    write_dna_file(primer_save_path, with_primer_seq_list)

if __name__ == '__main__':
    main_for_img() 
    # main_for_pdf()






