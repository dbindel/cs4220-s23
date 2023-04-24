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
	pdf/2023-03-24.pdf \
	pdf/2023-03-27.pdf \
	pdf/2023-03-29.pdf \
	pdf/2023-03-31.pdf \
	pdf/2023-04-10.pdf \
	pdf/2023-04-12.pdf \
	pdf/2023-04-14.pdf \
	pdf/2023-04-17.pdf \
	pdf/2023-04-19.pdf \
	pdf/2023-04-21.pdf 

PLUTO= \
	pluto/2023-01-23.jl \
	pluto/2023-01-25.jl \
	pluto/2023-02-03.jl \
	pluto/2023-02-08.jl \
	pluto/2023-02-10.jl \
	pluto/2023-02-13.jl \
	pluto/2023-02-20.jl \
	pluto/2023-02-22.jl \
	pluto/2023-02-24.jl \
	pluto/2023-03-06.jl \
	pluto/2023-03-08.jl \
	pluto/2023-03-13.jl \
	pluto/2023-03-15.jl \
	pluto/2023-03-20.jl \
	pluto/2023-03-22.jl \
	pluto/2023-03-24.jl \
	pluto/2023-03-27.jl \
	pluto/2023-03-29.jl \
	pluto/2023-03-31.jl \
	pluto/2023-04-10.jl \
	pluto/2023-04-12.jl \
	pluto/2023-04-14.jl \
	pluto/2023-04-17.jl \
	pluto/2023-04-19.jl \
	pluto/2023-04-21.jl 

HTML= \
	html/2023-01-23.html \
	html/2023-01-25.html \
	html/2023-02-03.html \
	html/2023-02-08.html \
	html/2023-02-10.html \
	html/2023-02-13.html \
	html/2023-02-20.html \
	html/2023-02-22.html \
	html/2023-02-24.html \
	html/2023-03-06.html \
	html/2023-03-08.html \
	html/2023-03-13.html \
	html/2023-03-15.html \
	html/2023-03-20.html \
	html/2023-03-22.html \
	html/2023-03-24.html \
	html/2023-03-27.html \
	html/2023-03-29.html \
	html/2023-03-31.html \
	html/2023-04-10.html \
	html/2023-04-12.html \
	html/2023-04-14.html \
	html/2023-04-17.html \
	html/2023-04-19.html \
	html/2023-04-21.html 

.PHONY: all copy fragments
all: $(LEC) $(HTML)

pdf/%.pdf: notes/%.tex
	latexmk --shell-escape -lualatex -outdir=pdf $<

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

pfigs:
	cd pluto ; \
	for f in *-tex.jl ; do \
		julia $$f ; \
	done

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/lec
	cp $(LEC) $(PLUTO) $(HTML) ../web/lec
