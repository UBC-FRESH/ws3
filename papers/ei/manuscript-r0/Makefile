# Simple LaTeX build with latexmk
PAPER := paper.tex
COVER := cover-letter.tex
EM_DIR := em-submission

.PHONY: all pdf cover clean distclean em-package

all: clean pdf cover

pdf:
	latexmk -pdf -interaction=nonstopmode -f $(PAPER)

cover:
	latexmk -pdf -interaction=nonstopmode -f $(COVER)

clean:
	# Remove aux files but keep the final PDF
	latexmk -c $(PAPER)
	latexmk -c $(COVER)

distclean:
	# Remove everything, including the final PDF
	latexmk -C $(PAPER)
	latexmk -C $(COVER)

em-package: pdf cover
	python scripts/prepare_em_submission.py
