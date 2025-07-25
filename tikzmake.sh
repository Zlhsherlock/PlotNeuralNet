# #!/bin/bash


# python $1.py 
# /usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode $1.tex

# rm *.aux *.log *.vscodeLog
# rm *.tex

# if [[ "$OSTYPE" == "darwin"* ]]; then
#     open $1.pdf
# else
#     xdg-open $1.pdf
# fi
#!/bin/bash
 
python $1.py 
pdflatex $1.tex
 
rm *.aux *.log *.vscodeLog
# rm *.tex  # 保留TEX文件
 
if [[ "$OSTYPE" == "darwin"* ]]; then
    open $1.pdf
else
    xdg-open $1.pdf
fi