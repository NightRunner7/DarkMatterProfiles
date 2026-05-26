#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess, re


# In[16]:


def run_sim(GALAXYMASS, EVOLUTIONTIME, CROSSSECTION):
    with open('RSIDM_template.m', 'r') as f:
        template_sript = f.read()
        template_sript = template_sript.replace('_GALAXYMASS_', str(GALAXYMASS))
        template_sript = template_sript.replace('_EVOLUTIONTIME_', str(EVOLUTIONTIME))
        template_sript = template_sript.replace('_CROSSSECTION_', str(CROSSSECTION))
        with open('RSIDM_run.m', 'w') as of:
            of.write(template_sript)
        process = subprocess.Popen('math -run "<<RSIDM_run.m"', stdout=subprocess.PIPE)
        for line in process.stdout:
            print(line.decode('utf8'), end='')


# In[ ]:


pat = r'ρsol_M_(\d+\.\d+)_t_(\d+\.\d+)_sigma_(\d+\.\d+)'

names = '''
ρsol_M_8.0_t_80900.0_sigma_0.01
'''

for name in names.split('\n')[-1]:
    print(name)

    match = re.match(pat, name)
    if match:
        print(match.groups())
        GALAXYMASS = match.group(1)
        EVOLUTIONTIME = match.group(2)
        CROSSSECTION = match.group(3)
        print(f'GALAXYMASS={GALAXYMASS}, EVOLUTIONTIME={EVOLUTIONTIME}, CROSSSECTION={CROSSSECTION}')
        run_sim(GALAXYMASS, EVOLUTIONTIME, CROSSSECTION)


# In[ ]: