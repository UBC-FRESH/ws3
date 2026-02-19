# Simple LaTeX build with latexmk
PAPER := paper.tex
COVER := cover-letter.tex
EM_DIR := em-submission
R0_DIR := ../manuscript-r0
DIFF_DIR := latexdiff-em

.PHONY: all pdf cover clean distclean em-package em-package-r0 latexdiff

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

em-package-r0:
	python $(R0_DIR)/scripts/prepare_em_submission.py

latexdiff: em-package em-package-r0
	mkdir -p $(DIFF_DIR)
	cp -a $(EM_DIR)/. $(DIFF_DIR)/
	PERLIO=:utf8 latexdiff --encoding=utf8 $(R0_DIR)/$(EM_DIR)/paper.tex $(EM_DIR)/paper.tex > $(DIFF_DIR)/paper.tex
	latexmk -pdf -interaction=nonstopmode -f -cd $(DIFF_DIR)/paper.tex
