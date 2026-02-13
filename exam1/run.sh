file='exam'

pdflatex -jobname=${file} '\def\nosol{True}\input' ${file}.tex
pdflatex -jobname=${file} '\def\nosol{True}\input' ${file}.tex

pdflatex -jobname=${file}_sol  ${file}.tex
pdflatex -jobname=${file}_sol  ${file}.tex

