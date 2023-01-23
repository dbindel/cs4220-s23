TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2023-01-23.pdf \
	pdf/2023-01-25.pdf \
	pdf/2023-01-27.pdf 

PLUTO= \
	pluto/2023-01-23.jl \
	pluto/2023-01-25.jl 

HTML= \
	html/2023-01-23.html \
	html/2023-01-25.html 

.PHONY: all copy fragments
all: $(LEC) $(HTML)

pdf/%.pdf: notes/%.tex
	latexmk -xelatex -outdir=pdf $<

html/%.html: pluto/%.jl
	../util/pluto2html.sh $<

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

fragments:
	for f in pluto/* ; do \
		../util/pluto2frag.py $$f code/ ; \
	done

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/lec
	cp $(LEC) $(PLUTO) $(HTML) ../web/lec
