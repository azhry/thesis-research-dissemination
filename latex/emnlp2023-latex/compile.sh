# 1. First pass to generate auxiliary files (.aux)
pdflatex emnlp2023.tex

# 2. Run bibtex to process the custom.bib file using the .aux file
bibtex emnlp2023.aux

# 3. Second pass to include the compiled bibliography into the document
pdflatex emnlp2023.tex

# 4. Final pass to resolve all cross-references correctly
pdflatex emnlp2023.tex