TEXINPUTS:=.

CLASS_HW=\
	pdf/2023-01-25-class.pdf \
	pdf/2023-01-27-class.pdf \
	pdf/2023-01-30-class.pdf \
	pdf/2023-02-01-class.pdf \
	pdf/2023-02-03-class.pdf \
	pdf/2023-02-06-class.pdf \
	pdf/2023-02-08-class.pdf \
	pdf/2023-02-10-class.pdf \
	pdf/2023-02-13-class.pdf \
	pdf/2023-02-15-class.pdf \
	pdf/2023-02-17-class.pdf \
	pdf/2023-02-20-class.pdf \
	pdf/2023-02-22-class.pdf \
	pdf/2023-02-24-class.pdf \
	pdf/2023-03-01-class.pdf \
	pdf/2023-03-03-class.pdf \
	pdf/2023-03-06-class.pdf \
	pdf/2023-03-08-class.pdf \
	pdf/2023-03-10-class.pdf \
	pdf/2023-03-13-class.pdf \
	pdf/2023-03-15-class.pdf \
	pdf/2023-03-17-class.pdf \
	pdf/proj-5223.pdf

HW=$(CLASS_HW) \
	pluto/hw1.jl html/hw1.html \
	pluto/hw2.jl html/hw2.html \
	pluto/proj1.jl html/proj1.html \
	pluto/hw3.jl html/hw3.html \
	pluto/midterm.jl html/midterm.html \
	pluto/hw4.jl html/hw4.html 

.PHONY: all copy
all: $(HW)

pdf/%.pdf: tex/%.tex | pdf
	xelatex -output-dir pdf $<
	xelatex -output-dir pdf $<

pdf: 
	@mkdir -p pdf

clean:
	rm -f pdf/*.aux pdf/*.log pdf/*.out
	rm -f pdf/*.nav pdf/*.snm pdf/*.toc 
	rm -f pdf/*.vrb

distclean: clean
	rm -f pdf/*.pdf

copy:
	mkdir -p ../web/hw
	cp $(HW) ../web/hw
