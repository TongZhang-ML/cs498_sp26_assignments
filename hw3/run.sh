#!/bin/bash
set -euo pipefail

cd latex

file='hw'
build_dir="build"

mkdir -p "${build_dir}"

# ---- PDFs (all aux/log/etc go to build/) ----
pdflatex -output-directory="${build_dir}" -jobname="${file}"     '\def\nosol{True}\input'"{${file}.tex}"
pdflatex -output-directory="${build_dir}" -jobname="${file}"     '\def\nosol{True}\input'"{${file}.tex}"

pdflatex -output-directory="${build_dir}" -jobname="${file}_sol" "${file}.tex"
pdflatex -output-directory="${build_dir}" -jobname="${file}_sol" "${file}.tex"

# ---- Student-facing release ----
/bin/rm -rf ../release
mkdir -p ../release/written_template_latex/questions

sed '/\\begin{solution}/,/\\end{solution}/d' "${file}.tex" > "../release/written_template_latex/${file}_template.tex"

cp myheaders.tex      ../release/written_template_latex/myheaders.tex
cp instructions.tex   ../release/written_template_latex/instructions.tex

# Copy the student PDF from build/
cp "${build_dir}/${file}.pdf" ../release/${file}.pdf

questionfilenames=$(ls questions/*.tex)
for questionfile in $questionfilenames
do
  sed '/\\begin{solution}/,/\\end{solution}/d' "${questionfile}" > "../release/written_template_latex/${questionfile}"
done

cp -R ../code/template ../release/code_template

