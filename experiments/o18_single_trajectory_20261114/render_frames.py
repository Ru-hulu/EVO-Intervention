"""Render exact, uniformly sampled source frames; no model/GT curves are synthesized."""
from pathlib import Path
import sys
import hashlib,json,subprocess
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from PIL import Image
from pypdf import PdfReader

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
PROV=json.loads((ROOT/'source/trajectory-provenance.json').read_text())
VIDEO=ROOT/'O18-validation-episode-000028.mp4'
assert hashlib.sha256(VIDEO.read_bytes()).hexdigest()==PROV['selected']['video_sha256']
N=PROV['selected']['total_frames']; FPS=PROV['selected']['fps']; COUNT=6
# Round exact evenly spaced locations over the entire source trajectory, endpoints included.
INDICES=[round(i*(N-1)/(COUNT-1)) for i in range(COUNT)]
assert INDICES[0]==0 and INDICES[-1]==N-1
assert max(b-a for a,b in zip(INDICES,INDICES[1:]))-min(b-a for a,b in zip(INDICES,INDICES[1:]))<=1
frames=ROOT/'source/frames';frames.mkdir(exist_ok=True)
select='select='+ '+'.join(f'eq(n\\,{i})' for i in INDICES)
subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(VIDEO),'-vf',select,'-fps_mode','vfr','-frames:v',str(COUNT),str(frames/'frame-%02d.png')],check=True)
paths=sorted(frames.glob('frame-*.png'));assert len(paths)==COUNT
sizes=[Image.open(p).size for p in paths];assert all(x==(1280,720) for x in sizes)
WIDTH=16.8*cm; GAP=1.5; LABEL=12
PANEL=(WIDTH-GAP*(COUNT-1))/COUNT; IMAGE_HEIGHT=PANEL*720/1280; HEIGHT=IMAGE_HEIGHT+LABEL
PDF=ROOT/'output/pdf/O18-episode-000028-uniform-6-frames.pdf'
c=canvas.Canvas(str(PDF),pagesize=(WIDTH,HEIGHT),pageCompression=1)
c.setTitle('O18 validation episode 000028 - six uniformly sampled frames')
c.setAuthor('ProPriMem experiment visualization')
for i,(p,frame_index) in enumerate(zip(paths,INDICES)):
 x=i*(PANEL+GAP)
 c.drawImage(ImageReader(str(p)),x,LABEL,width=PANEL,height=IMAGE_HEIGHT,mask='auto')
 c.setFont('Times-Roman',7.2);c.setFillColorRGB(.13,.13,.13)
 c.drawCentredString(x+PANEL/2,3.6,f'({chr(97+i)}) {frame_index/FPS:.1f} s')
c.showPage();c.save()
reader=PdfReader(PDF);assert len(reader.pages)==1
page=reader.pages[0];assert abs(float(page.mediabox.width)-WIDTH)<.01
subprocess.run(['pdftoppm','-png','-r','450','-singlefile',str(PDF),str(ROOT/'O18-episode-000028-uniform-6-frames')],check=True)
receipt=dict(status='passed',sample_id=PROV['selected']['sample_id'],split='val',video_sha256=PROV['selected']['video_sha256'],source_frame_count=N,source_fps=FPS,selected_frame_indices_zero_based=INDICES,selected_timestamps_seconds=[i/FPS for i in INDICES],selection='uniform over complete trajectory including first and last frames',main_view_only=True,crop_applied=False,frame_count=COUNT,pdf_width_cm=WIDTH/cm,pdf_height_cm=HEIGHT/cm,curve_status='not_generated_no_existing_predictions',frame_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(ROOT/'source/figure-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
(ROOT/'figure.tex').write_text('''% Place the PDF beside your LaTeX manuscript. Requires \\usepackage{graphicx}.
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{O18-episode-000028-uniform-6-frames.pdf}
  \\caption{Six uniformly sampled observations from a held-out O18 trajectory (episode 000028). The task is to inspect three spray bottles from left to right, returning each bottle before proceeding to the next.}
  \\label{fig:o18-trajectory}
\\end{figure}
''')
print(json.dumps(receipt,indent=2))
