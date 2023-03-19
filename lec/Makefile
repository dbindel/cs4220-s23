TEXINPUTS:=.

LEC= \
	pdf/background.pdf \
	pdf/2023-01-23.pdf \
	pdf/2023-01-25.pdf \
	pdf/2023-01-27.pdf \
	pdf/2023-01-30.pdf \
	pdf/2023-02-01.pdf \
	pdf/2023-02-03.pdf \
	pdf/2023-02-08.pdf \
	pdf/2023-02-10.pdf \
	pdf/2023-02-13.pdf \
	pdf/2023-02-15.pdf \
	pdf/2023-02-17.pdf \
	pdf/2023-02-20.pdf \
	pdf/2023-02-22.pdf \
	pdf/2023-02-24.pdf \
	pdf/2023-03-01.pdf \
	pdf/2023-03-03.pdf \
	pdf/2023-03-06.pdf \
	pdf/2023-03-08.pdf \
	pdf/2023-03-13.pdf \
	pdf/2023-03-15.pdf \
	pdf/2023-03-17.pdf \
	pdf/2023-03-20.pdf \
	pdf/2023-03-22.pdf \
	pdf/2023-03-24.pdf

PLUTO= \
	pluto/2023-01-23.jl \
	pluto/2023-01-25.jl \
	pluto/2023-02-03.jl \
	pluto/2023-02-08.jl \
	pluto/2023-02-10.jl 

HTML= \
	html/2023-01-23.html \
	html/2023-01-25.html \
	html/2023-02-03.html \
	html/2023-02-08.html \
	html/2023-02-10.html 

.PHONY: all copy fragments
all: $(LEC) $(HTML)

pdf/%.pdf: notes/%.tex
	latexmk -xelatex -outdir=pdf $<

# html/%.html: pluto/%.jl
# 	../util/pluto2html.sh $<

clean:
	rm -f pdf/*.bbl pdf/*.blg pdf/*.fls
	rm -f pdf/*.fdb_latexmk pdf/*.xdv
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
