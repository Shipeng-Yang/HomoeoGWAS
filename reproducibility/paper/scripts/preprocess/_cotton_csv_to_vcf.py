#!/usr/bin/env python3
"""hebau cotton CSV -> bgzipped VCF (single chromosome).

CSV 列: CHROM,POS,REF,ALT,maf,B001,B002,...
基因型字符串: "TT"/"TC"/"--"(缺失)  或半缺失 "T-".

用法: python _cotton_csv_to_vcf.py <chr_A01.csv.gz> <chr_A01.vcf.gz> <CHROM_id>
"""
import csv
import gzip
import os
import subprocess
import sys


def encode(g: str, ref: str, alt: str) -> str:
    if len(g) != 2 or '-' in g or 'N' in g.upper():
        return './.'
    a, b = g[0], g[1]
    code = []
    for x in (a, b):
        if x == ref:
            code.append('0')
        elif x == alt:
            code.append('1')
        else:
            return './.'
    return f'{code[0]}/{code[1]}'


def main(csv_path: str, out_vcf: str, chrom_id: str) -> None:
    tmp_vcf = out_vcf + '.tmp'
    with gzip.open(csv_path, 'rt') as f, open(tmp_vcf, 'w') as o:
        r = csv.reader(f)
        header = next(r)
        samples = header[5:]
        o.write('##fileformat=VCFv4.2\n')
        o.write(f'##contig=<ID={chrom_id}>\n')
        o.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        o.write('\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT',
                           'QUAL', 'FILTER', 'INFO', 'FORMAT'] + samples) + '\n')
        for row in r:
            if not row:
                continue
            pos, ref, alt = row[1], row[2], row[3]
            gts = row[5:]
            rec = [chrom_id, pos, '.', ref, alt, '.', '.', '.', 'GT']
            rec += [encode(g, ref, alt) for g in gts]
            o.write('\t'.join(rec) + '\n')
    # bgzip
    subprocess.run(['bgzip', '-f', tmp_vcf], check=True)
    os.replace(tmp_vcf + '.gz', out_vcf)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('usage: _cotton_csv_to_vcf.py <in.csv.gz> <out.vcf.gz> <chrom_id>')
    main(sys.argv[1], sys.argv[2], sys.argv[3])
