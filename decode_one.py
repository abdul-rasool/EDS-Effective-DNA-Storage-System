import struct

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

# decode rule ID
def rule_decode(bases):
    binary_str = ""
    decode_rule = [{'A':'0','T':'1'},{'C':'0','G':'1'},{'A':'0','T':'1'},{'C':'0','G':'1'}]
    for i in range(4):
        binary_str += decode_rule[i][bases[i]]
    rule = int(binary_str,2)
    return rule

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

# read DNA file
def read_dna_file(path):
    dna_sequences = []
    with open(path, "r") as file:
        for seq in file:
            if seq.startswith(">"):
                continue
            seq = seq.strip()[20:]
            seq_list = list(seq[:-20])
            dna_sequences.append(seq_list)
        # # Read row by row
        # lines = file.readlines()
        # for  index, line in enumerate(lines):
        #     dna_sequences.append(list(line.replace("\n", "")))
    return dna_sequences

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

def part_index(part):
    encode_rule = {'00':'A','01':'T','10':'C','11':'G'}
    binary = str(bin(part))[2:].zfill(4)
    result = encode_rule[binary[:2]] + encode_rule[binary[2:]]
    return  result

def decode_part(id,data):
    part_data = []
    part_bases = part_index(id)
    data_length = len(data)
    for i in range(data_length):
        if data[i][0]+data[i][1] == part_bases:
            part_data.append(data[i])
    part_data_length = len(part_data)
    without_partID = []
    for j in range(part_data_length):
        without_partID.append(part_data[j][2:])
    return without_partID

def decode_for_img():
    # original_size we know this time, It can also encode this value(meta data) to sequence
    meta_data = [3092, 5770, 5741, 2652, 5802, 6290, 6331, 5985, 7051, 7118, 6551, 5802, 3011, 5875, 5903, 2536]
    part_id = 0
    original_size = meta_data[part_id]

    input_path = "./imageResults/result.dna"
    d = read_dna_file(input_path)
    # delete primers
    without_primes = []
    data_len = len(d)
    for i in range(data_len):
        without_primes.append(d[i][20:-20])

    part = decode_part(part_id, without_primes)
    m = decode(part, B_set=B_set, E_set=E_set)
    # If there is no error, delete the redundancy directly
    without_redundancy = []
    if len(m) % 3 == 0:
        for i in range(len(m)):
            if (i + 1) % 3 != 0:
                without_redundancy.append(m[i])
    else:
        for i in range(len(m) - 2):
            if (i + 1) % 3 != 0:
                without_redundancy.append(m[i])
        without_redundancy.append(m[-2])
    write_BinaryMatrix_to_file('mri_'+str(part_id)+'.jpg', without_redundancy, original_size)

def decode_for_pdf():
    # No need to divide into parts
    origin_size = 57936
    input_path = './reportResults/result.dna'
    # record the origin size
    d = read_dna_file(input_path)
    # delete primers
    without_primes = []
    data_len = len(d)
    for i in range(data_len):
        without_primes.append(d[i][20:-20])

    m = decode(without_primes, B_set=B_set, E_set=E_set)
    without_redundancy = []
    if len(m) % 3 == 0:
        for i in range(len(m)):
            if (i + 1) % 3 != 0:
                without_redundancy.append(m[i])
    else:
        for i in range(len(m) - 2):
            if (i + 1) % 3 != 0:
                without_redundancy.append(m[i])
        without_redundancy.append(m[-2])
    write_BinaryMatrix_to_file('mri.pdf', without_redundancy, origin_size)

if __name__ == '__main__':
    # decode_for_img()
    decode_for_pdf()

