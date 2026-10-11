import subprocess, shlex
AD = '/content/adapters/Qwen_2.5_7B_instruct/report_dataset_full_1684p_20260831-133510'
PL = 'pleasure=npz:/content/work/Qwen_2.5_7B_instruct/analysis/vectors.npz:Lpleasure|pleasure:s2:24'
PA = 'pain=pt:/content/Pain-axis/results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt'
args = ['python', 'feelprobe.py', '--adapter', AD, '--layer', '16', '--vector', PL, '--vector', PA,
        '--doses', '0.5,1,1.5,2,2.5,3,4,6', '--tag', 'q7_adapter']
cmd = 'cd /content/pleasure/code && nohup ' + ' '.join(shlex.quote(a) for a in args) + ' > /content/work/feelprobe_q7.out 2>&1 &'
subprocess.Popen(cmd, shell=True)
print('launched:', cmd[:200])
