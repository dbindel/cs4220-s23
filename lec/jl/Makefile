# ../../util/plutexify.jl 2023-03-29.jl -P preamble.tex -s -t 2023-03-29.tex -p notebook.jl

PLUTEXIFY=../../util/plutexify.jl

SRC=	2023-03-29.jl \
	2023-03-31.jl \
	2023-04-10.jl \
	2023-04-12.jl \
	2023-04-14.jl \
	2023-04-17.jl \
	2023-04-19.jl \
	2023-04-21.jl \
	2023-04-24.jl \
	2023-04-26.jl \
	2023-04-28.jl

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
