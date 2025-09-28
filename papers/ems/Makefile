# Simple LaTeX build with latexmk
PAPER := paper.tex

.PHONY: all pdf clean distclean

all: clean pdf

pdf:
	latexmk -pdf -interaction=nonstopmode -f $(PAPER)

clean:
	# Remove aux files but keep the final PDF
	latexmk -c $(PAPER)

distclean:
	# Remove everything, including the final PDF
	latexmk -C $(PAPER)