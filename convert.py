import re
import sys

def convert_md_to_latex(md_text):
    latex = []
    lines = md_text.split('\n')
    
    in_code_block = False
    in_itemize = False
    in_table = False
    table_lines = []
    
    def process_table(t_lines):
        if not t_lines: return []
        out = []
        out.append('\\begin{table}[ht]')
        out.append('\\centering')
        
        sep_line = t_lines[1] if len(t_lines) > 1 else ""
        cols = sep_line.count('|') - 1 if '|' in sep_line else 1
        col_format = 'l' * cols
        
        out.append(f'\\begin{{tabular}}{{{col_format}}}')
        out.append('\\toprule')
        
        for i, line in enumerate(t_lines):
            if set(line.strip().replace('|', '').replace('-', '').replace(':', '')) == set():
                out.append('\\midrule')
                continue
            
            row = [c.strip() for c in line.strip().split('|')][1:-1]
            if not row: continue
            
            row = [re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', c) for c in row]
            row = [re.sub(r'`(.*?)`', r'\\texttt{\1}', c) for c in row]
            row = [re.sub(r'\[(.*?)\]\(.*?\)', r'\1', c) for c in row] # Strip links in tables
            row = [c.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#') for c in row]
            
            out.append(' & '.join(row) + ' \\\\')
            
        out.append('\\bottomrule')
        out.append('\\end{tabular}')
        out.append('\\caption{Technical details.}')
        out.append('\\end{table}')
        return out

    for line in lines:
        if line.startswith('```'):
            if in_code_block:
                latex.append('\\end{verbatim}')
                in_code_block = False
            else:
                latex.append('\\begin{verbatim}')
                in_code_block = True
            continue
            
        if in_code_block:
            latex.append(line)
            continue
            
        if line.strip().startswith('|') and line.strip().endswith('|'):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            latex.extend(process_table(table_lines))
            in_table = False
            table_lines = []
            
        if line.startswith('# '):
            if in_itemize: latex.append('\\end{itemize}'); in_itemize=False
            latex.append('\\chapter{' + line[2:].strip().replace('_', '\\_').replace('#', '\\#') + '}')
            continue
        elif line.startswith('## '):
            if in_itemize: latex.append('\\end{itemize}'); in_itemize=False
            latex.append('\\section{' + line[3:].strip().replace('_', '\\_').replace('#', '\\#') + '}')
            continue
        elif line.startswith('### '):
            if in_itemize: latex.append('\\end{itemize}'); in_itemize=False
            latex.append('\\subsection{' + line[4:].strip().replace('_', '\\_').replace('#', '\\#') + '}')
            continue
        elif line.startswith('#### '):
            if in_itemize: latex.append('\\end{itemize}'); in_itemize=False
            latex.append('\\subsubsection{' + line[5:].strip().replace('_', '\\_').replace('#', '\\#') + '}')
            continue
            
        list_match = re.match(r'^(\s*)[-*]\s+(.*)', line)
        if list_match:
            if not in_itemize:
                latex.append('\\begin{itemize}')
                in_itemize = True
            item_text = list_match.group(2)
            item_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', item_text)
            item_text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', item_text)
            item_text = re.sub(r'`(.*?)`', r'\\texttt{\1}', item_text)
            item_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', item_text) # strip links
            item_text = item_text.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
            latex.append('    \\item ' + item_text)
            continue
        else:
            if in_itemize and line.strip() == '':
                pass
            elif in_itemize and not re.match(r'^\s*[-*]\s+(.*)', line):
                latex.append('\\end{itemize}')
                in_itemize = False
                
        num_match = re.match(r'^(\s*)\d+\.\s+(.*)', line)
        if num_match:
            item_text = num_match.group(2)
            item_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', item_text)
            item_text = re.sub(r'`(.*?)`', r'\\texttt{\1}', item_text)
            item_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', item_text) # strip links
            item_text = item_text.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
            latex.append('\\noindent $\\bullet$ ' + item_text)
            continue

        if line.strip() == '':
            latex.append('')
            continue
            
        if line.startswith('> '):
            latex.append('\\begin{quote}')
            l = line[2:]
            l = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', l)
            l = re.sub(r'`(.*?)`', r'\\texttt{\1}', l)
            l = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', l) # strip links
            l = l.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
            latex.append(l)
            latex.append('\\end{quote}')
            continue
            
        l = line
        l = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', l)
        l = re.sub(r'\*(.*?)\*', r'\\textit{\1}', l)
        l = re.sub(r'`(.*?)`', r'\\texttt{\1}', l)
        l = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', l) # strip links
        l = l.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#')
        
        latex.append(l)
        
    if in_itemize:
        latex.append('\\end{itemize}')
    if in_table:
        latex.extend(process_table(table_lines))
        
    latex_text = '\n'.join(latex)
    
    def unescape_math(m):
        return m.group(0).replace('\\\\_', '_').replace('\\_', '_').replace('\\#', '#')
        
    latex_text = re.sub(r'\$\$.*?\$\$', unescape_math, latex_text, flags=re.DOTALL)
    latex_text = re.sub(r'\$.*?\$', unescape_math, latex_text, flags=re.DOTALL)
    
    return latex_text

if __name__ == "__main__":
    with open(r'c:\Users\user\Desktop\ubi\research\thesis\experiments\experiment-5-unified-model\technical_appendix_exp5.md', 'r', encoding='utf-8') as f:
        md = f.read()

    latex = convert_md_to_latex(md)
    latex = latex.replace('\\chapter{Technical Documentation: Unified Multimodal Graph-Gated Mixture-of-Experts (Experiment 5)}', '\\chapter{Technical Appendix: Unified Multimodal Graph-Gated MoE (Experiment 5)}\\label{app:experiment5_tech}')
    
    with open(r'c:\Users\user\Desktop\ubi\research\thesis\master-thesis\appendix_experiment_5.tex', 'w', encoding='utf-8') as f:
        f.write(latex)
    print("Conversion complete!")
