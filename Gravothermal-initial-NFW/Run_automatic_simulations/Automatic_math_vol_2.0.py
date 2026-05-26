#!/usr/bin/env python
# coding: utf-8

import subprocess, re

def run_sim(GALAXYMASS, EVOLUTIONTIME, CROSSSECTION):
    with open('RSIDM_template.m', 'r') as f:
        print("ARG:", GALAXYMASS, EVOLUTIONTIME, CROSSSECTION)
        template_sript = f.read()
        template_sript = template_sript.replace('GALAXYMASS', str(GALAXYMASS))
        template_sript = template_sript.replace('EVOLUTIONTIME', str(EVOLUTIONTIME))
        template_sript = template_sript.replace('CROSSSECTION', str(CROSSSECTION))
        # print(template_sript)
        with open('RSIDM_run.m', 'w') as of:
            of.write(template_sript)
        process = subprocess.Popen('math -run "<<RSIDM_run.m"', stdout=subprocess.PIPE)
        for line in process.stdout:
            print(line.decode('utf8'), end='')


pat = r'ρsol_M_(\d+\.\d+)_t_(\d+\.\d+)_sigma_(\d+\.\d+)'

names = '''
ρsol_M_11.30103_t_10.0_sigma_0.1
ρsol_M_11.30103_t_10.0_sigma_0.2
ρsol_M_11.30103_t_10.0_sigma_0.5
ρsol_M_11.30103_t_10.0_sigma_1.0
ρsol_M_11.30103_t_10.0_sigma_2.0
ρsol_M_11.30103_t_10.0_sigma_5.0
ρsol_M_11.30103_t_10.0_sigma_10.0
ρsol_M_11.30103_t_10.0_sigma_20.0
ρsol_M_11.30103_t_10.0_sigma_50.0
ρsol_M_11.30103_t_10.0_sigma_100.0
ρsol_M_11.30103_t_10.0_sigma_200.0
ρsol_M_11.30103_t_10.0_sigma_500.0
ρsol_M_11.30103_t_10.0_sigma_1000.0
'''

# List containg all split names -> every element is coresponding to name of one file
print(names.split('\n'))

list_names = names.split('\n')
del list_names[0]
del list_names[-1]

print("list_name:", list_names)

for name in list_names:

    match = re.match(pat, name)

    if match:
        print()
        print(match.groups())
        GALAXYMASS = match.group(1)
        EVOLUTIONTIME = match.group(2)
        CROSSSECTION = match.group(3)
        print(f'GALAXYMASS={GALAXYMASS}, EVOLUTIONTIME={EVOLUTIONTIME}, CROSSSECTION={CROSSSECTION}')
        run_sim(GALAXYMASS, EVOLUTIONTIME, CROSSSECTION)
