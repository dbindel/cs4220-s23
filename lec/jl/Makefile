# ../../util/plutexify.jl 2023-03-29.jl -P preamble.tex -s -t 2023-03-29.tex -p notebook.jl

PLUTEXIFY=../../util/plutexify.jl

SRC=2023-03-29.jl
TEX=$(patsubst %.jl,../notes/%.tex,$(SRC))
PLUTO=$(patsubst %.jl,../pluto2/%.jl,$(SRC))

../notes/%.tex: %.jl
	julia $<
	$(PLUTEXIFY) -P preamble.tex -s -t $@ $<

../pluto2/%.jl: %.jl
	$(PLUTEXIFY) -p $@ $<

.PHONY: tex pluto all

tex: $(TEX)
pluto: $(PLUTO)
all: $(TEX) $(PLUTO)