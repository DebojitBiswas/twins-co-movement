#!/usr/bin/env python3
"""Build the NMI-style submission package from the working draft (paper/*.tex): overleaf/main.tex (main text, Methods, Extended Data =
ED Table 1, ED Table 2, ED Fig. 1) and overleaf/supplement.tex (Supplementary Tables 1-9 = former ED Tables 3-11; Supplementary Notes = the
former ED figure descriptions).  Cross-references are remapped.  Then compiles both with tectonic and zips the project."""
import re, os, shutil, subprocess
B = 'paper/'; O = 'overleaf/'
def inline(t):
    return re.sub(r'^\\input\{([^}]+)\}\s*$', lambda m: inline(open(B + m.group(1) + '.tex').read()), t, flags=re.M)

def remap(t):
    t = re.sub(r'Extended Data Table~(\d+)', lambda m: m.group(0) if int(m.group(1)) <= 2 else f'Supplementary Table~{int(m.group(1)) - 2}', t)
    t = t.replace('Extended Data Fig.~6', 'Extended Data Fig.~1')
    t = re.sub(r'Extended Data Fig\.~(\d+)', lambda m: m.group(0) if m.group(1) == '1' else f'Supplementary Note~{m.group(1)}', t)
    t = t.replace('reported in Extended Data as such', 'reported in Supplementary Note~2 as such')
    t = re.sub(r'Extended Data(?! Table| Fig| \\)', 'Supplementary Information', t)
    t = re.sub(r'ED Table (\d+)', lambda m: m.group(0) if int(m.group(1)) <= 2 else f'Supplementary Table {int(m.group(1)) - 2}', t)
    return t

main = open(B + 'main_v1.tex').read().replace('\\graphicspath{{../figs/}}', '\\graphicspath{{figs/}}')
main = re.sub(r'^% Paper C — draft v1 skeleton.*?\n% c7_v1/summary\.json.*?\n', '', main, flags=re.S)
pre, body = main.split('\\begin{document}')
body = inline(body)
# --- split the Extended Data section
ed_start = body.index('% Extended Data — v1'); ed_end = body.index('\\bibliographystyle')
ed = body[ed_start:ed_end]; before = body[:ed_start]; after = body[ed_end:]
paras = re.split(r'(?=\\paragraph\{ED )', ed)
keep, supp_tables, supp_notes, edfig = [], [], [], ''
for p in paras:
    if re.match(r'\\paragraph\{ED Table [12] ', p): keep.append(p)
    elif p.startswith('\\paragraph{ED Table'): supp_tables.append(p)
    elif p.startswith('\\paragraph{ED Figure'):
        if '\\begin{figure}' in p:   # the drawn figure (inverse-controller trajectories) travels with the main text as ED Fig. 1
            fig = p[p.index('\\begin{figure}'):p.index('\\end{figure}') + len('\\end{figure}')]; p = p.replace(fig, '')
            edfig = fig.replace('ED Figure 6 ---', 'Extended Data Figure 1 ---')
        supp_notes.append(p)
keep_txt = ''.join(keep).replace('\\paragraph{ED Table 1', '\\paragraph{Extended Data Table 1').replace('\\paragraph{ED Table 2', '\\paragraph{Extended Data Table 2')
main_body = remap(before) + '\\section*{Extended Data}\n' + remap(keep_txt) + '\n' + remap(edfig) + '\n' + remap(after)
main_tex = remap(pre) + '\\begin{document}' + main_body
# --- supplement
supp = '\n'.join(remap(p).replace('\\paragraph{ED Table', '\\paragraph{Supplementary Table') for p in supp_tables)
supp_notes.sort(key=lambda p: int(re.search(r'ED Figure (\d+)', p).group(1)))
notes = '\n'.join(remap(p).replace('\\paragraph{ED Figure', '\\paragraph{Supplementary Note') for p in supp_notes)
notes = notes.replace('\\paragraph{Supplementary Note 7 ---', '\\paragraph{Supplementary Note 6 --- Inverse-controller test, full trajectories.} See Extended Data Fig.~1.\n\\paragraph{Supplementary Note 7 ---')
supp_pre = remap(pre).replace('\\title{', '\\title{Supplementary Information for: ')
full = open(B + 'results_full.tex').read().replace('\\section*{Results}\n', '')
note12 = '\\paragraph{Supplementary Note 12 --- Extended results.} The full account of the Results, with every comparator, horizon and control; the main text condenses it.\n' + remap(full).replace('\\input{table_v1}', '')
supp_tex = supp_pre + '\\begin{document}\n\\maketitle\n\\section*{Supplementary Tables}\n' + supp + '\n\\section*{Supplementary Notes}\n' + notes + '\n' + note12 + '\n\\bibliographystyle{plainnat}\n\\bibliography{refs}\n\\end{document}\n'
os.makedirs(O + 'figs', exist_ok=True)
open(O + 'main.tex', 'w').write(main_tex); open(O + 'supplement.tex', 'w').write(supp_tex)
shutil.copy(B + 'refs.bib', O)
for f in re.findall(r'includegraphics\[[^]]*\]\{([^}]+)\}', main_tex + supp_tex): shutil.copy('figs/' + f, O + 'figs/')
open(O + 'README.txt', 'w').write('NMI-style package built by build_submission.py. main.tex = main text + Methods + Extended Data (Table 1, Table 2, Fig. 1); supplement.tex = Supplementary Tables 1-9 and Supplementary Notes. pdfLaTeX; natbib/plainnat; figs/ PNG.\n')
for f in ('main', 'supplement'):
    r = subprocess.run([os.path.abspath('tex_env/bin/tectonic'), '-X', 'compile', f + '.tex'], cwd=O, capture_output=True, text=True)
    errs = [l for l in r.stderr.splitlines() if l.startswith('error')]
    print(f, 'compiled' if not errs else errs[:3])
    if os.path.exists(O + f + '.pdf'): shutil.copy(O + f + '.pdf', f'pdf/paperC_{f}.pdf')
for f in ('main.pdf', 'supplement.pdf'):
    if os.path.exists(O + f): os.remove(O + f)
if os.path.exists('paperC_overleaf.zip'): os.remove('paperC_overleaf.zip')
subprocess.run(['zip', '-qr', 'paperC_overleaf.zip', 'overleaf']); print('zipped paperC_overleaf.zip')
