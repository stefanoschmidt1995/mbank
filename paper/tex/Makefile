all: mbank.pdf

mbank.pdf: mbank.tex mbank.bbl mbank.aux
	pdflatex -synctex=1 -interaction=nonstopmode "mbank".tex

mbank.aux:
	pdflatex -synctex=1 -interaction=nonstopmode "mbank".tex

mbank.bbl: mbank.aux
	bibtex "mbank".aux

clean:
	rm -f mbank.bbl mbank.out mbankNotes.bib mbank.aux mbank.log mbank.blg mbank.toc mbank.synctex.gz mbank.pdf
