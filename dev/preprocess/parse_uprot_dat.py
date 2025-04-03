import os, sys
from pathlib import Path
import gzip
import re

# # Step 1: Download the file
# url = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/uniprot_trembl_human.dat.gz'
# response = requests.get(url)
# with open('uniprot_trembl_human.dat.gz', 'wb') as f:
#     f.write(response.content)

# Step 2: Decompress and parse the file
def parse_uniprot_dat(file_path):
    with gzip.open(file_path, 'rt') as f:
        current_entry = {}
        function_lines = []
        inside_function = False
        current_entry['AC'] = []
        for line in f:
            line = line.rstrip()

            if line.startswith('//'):  # End of an entry
                if function_lines:
                    tmp_text = ' '.join(function_lines).replace('CC       ', '').replace('-!- FUNCTION: ', '')
                    tmp_text = re.sub(' \(PubMed:\d+(, PubMed:\d+)*\)', '', tmp_text).strip()
                    split_by_sentence = tmp_text.strip('.').split('. ')
                    split_by_sentence[-1] = re.sub(r'\{.*?\}$', '', split_by_sentence[-1]).strip()
                    function_text = '. '.join(split_by_sentence).strip('. ')
                    current_entry['Function'] = function_text

                yield current_entry

                current_entry = {}
                current_entry['AC'] = []
                function_lines = []
                inside_function = False

            elif line.startswith('ID'):
                info = line.split()
                if len(info) < 5:
                    print(info)
                current_entry['ID'] = info[1]
                current_entry['length'] = info[3]

            elif line.startswith('AC'):
                current_entry['AC'].extend([pid.strip(';') for pid in line.split()[1:]])

            elif line.startswith('DE'):
                # if 'Name' not in current_entry:
                name = line[5:].strip().rstrip('.')
                if name.startswith('RecName:') and re.search('Full=', name):
                    if 'Name' in current_entry:
                        continue
                    current_entry['Name'] = name.split('=')[1].rstrip(';').split('{')[0].strip()

                # else:
                #     current_entry['Name'] += " " + line[5:].strip().rstrip('.')
            elif line.startswith('GN') and re.search('Name=', line):
                # if 'Gene' not in current_entry:
                #     current_entry['Gene'] = line[5:].split('=')[1].strip().rstrip('.;')
                # else:
                #     current_entry['Gene'] += " " + line[5:].split('=')[1].strip().rstrip('.;')
                try:
                    current_entry['Gene'] = re.search(r'Name=(\S+)\b', line[5:].split('{')[0]).group(1)
                except:
                    print(line)
                    current_entry['Gene'] = ''
                
            elif line.startswith('DR') and 'GO;' in line:
                go_id = line.split(';')[1].strip()
                if 'GO' not in current_entry:
                    current_entry['GO'] = []
                current_entry['GO'].append(go_id)
            elif line.startswith('CC'):
                if '-!- FUNCTION:' in line:
                    inside_function = True
                    func_info = line.split('-!- FUNCTION:')[1].strip()
                    function_lines.append(func_info)
                elif inside_function:
                    if line.startswith('CC       ') or not line.startswith('CC   -!-'):
                        func_info = line.split(maxsplit=1)[1].strip()
                        function_lines.append(line)
                    else:
                        inside_function = False

if __name__ == '__main__':
    # data_root = Path('/local/storage/yl986/data/UniProt')
    data_root = Path('/home/yl986/data/UniProt')
    # input_fpath = data_root / 'uniprot_sprot_human.dat.gz'
    # output_fpath = data_root / 'uniprot_sprot_human_meta.txt'
    input_fpath = data_root / 'uniprot_trembl_human.dat.gz'
    output_fpath = data_root / 'uniprot_trembl_human_meta.txt'
    
    header = ['UniProt', 'ID', 'name', 'length', 'gene', 'function']
    with open(output_fpath, 'w') as fo:
        fo.write('\t'.join(header) + '\n')
        for entry in parse_uniprot_dat(input_fpath):
            pid = entry['ID']
            name = entry.get('Name', '')
            gene = entry.get('Gene', '')
            func_text = entry.get('Function', '')

            for uprot in entry['AC']:
                fo.write('\t'.join([uprot, pid, name, entry['length'], gene, func_text]) + '\n')
            # fo.write('\t'.join([entry['ID']]))
            # print(f"ID: {entry.get('ID', 'N/A')}")
            # print(f"Name: {entry.get('Name', 'N/A')}")
            # print(f"Gene: {entry.get('Gene', 'N/A')}")
            # print(f"GO: {', '.join(entry.get('GO', []))}")
            # print('---')
