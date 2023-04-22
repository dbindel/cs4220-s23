#!/bin/bash
#=
exec julia --color=yes --startup-file=no -e 'include(popfirst!(ARGS))' \
    "${BASH_SOURCE[0]}" "$@"
=#

using Pluto
using Markdown
using Mustache


# --- Lexical analysis ---

# We treat the input file as a stream of lines (one line is one token).
# Standard comment lines and white space lines are ignored, but we identify
# triple quotes and block comment begin/end in order to identify text blocks,
# standard code constructs matching with end (and end lines) to identify
# code blocks, and specially formatted comments for starting/ending cells
# where this strategy doesn't work and for setting global mode parameters

# Line types
@enum LTokType begin
    TQ=1         # Triple quote
    WS=2         # White space only
    BEGIN=3      # Begin block (begin, let, function, for, try, struct, while)
    BEND=4       # End block (left justified)
    CLINE=5      # Comment line
    CSTART=6     # Start multi-line comment
    CEND=7       # End multi-line comment
    MCELL=8      # Cell meta-line     (#-cell)
    MEND=9       # Cell end meta-line (#-end)
    MDCELL=10    # Disabled cell      (#=cell)
    MODE=11      # Mode flags line    (#-mode)
    CIF=12       # Disabled if block  (#=if)
    MIF=13       # Meta-line if block (#-if)
    GENERAL=14   # General line
end

function Base.show(io :: IO, tok :: LTokType)
    names = ["TQ", "WS", "BEGIN", "END", "CLINE",
             "CSTART", "CEND", "MCELL", "MEND", "MDCELL",
             "MODE", "CIF", "MIF", "GENERAL"]
    names[Int(tok)]
end

function toktype(line)
    if     occursin(r"^\"\"\"", line)        TQ
    elseif occursin(r"^\s*$",   line)        WS
    elseif occursin(r"^let(\s+|$)", line)       BEGIN
    elseif occursin(r"^begin(\s+|$)", line)     BEGIN
    elseif occursin(r"^function(\s+|$)", line)  BEGIN
    elseif occursin(r"^for(\s+|$)", line)       BEGIN
    elseif occursin(r"^while(\s+|$)", line)     BEGIN
    elseif occursin(r"^try(\s+|$)", line)       BEGIN
    elseif occursin(r"^(mutable\s+)?struct(\s+|$)", line)  BEGIN
    elseif occursin(r"^end(^\s+|$)", line)   BEND
    elseif occursin(r"^#=cell", line)        MDCELL
    elseif occursin(r"^#=if", line)          CIF
    elseif occursin(r"^#=", line)            CSTART
    elseif occursin(r"^=#", line)            CEND
    elseif occursin(r"^#-cell", line)        MCELL
    elseif occursin(r"^#-end", line)         MEND
    elseif occursin(r"^#-mode", line)        MODE
    elseif occursin(r"^#-if", line)          MIF
    elseif occursin(r"^#",  line)            CLINE
    else                                     GENERAL
    end
end


# --- Cells ---

# A cell is a block of text or code along with some metadata flags,
# scraped together from the input file.

struct Cell
    contents :: String
    flags :: Dict{String,String}
    is_code :: Bool
end

code_cell(contents, flags) = Cell(contents, flags, true)
text_cell(contents, flags) = Cell(contents, flags, false)

function Base.show(io :: IO, c :: Cell)
    if (!c.is_code)
        print(io, "#= $(c.flags)\n")
        print(io, "$(c.contents)\n=#")
    else
        print(io, "# -- code block: $(c.flags)\n")
        print(io, "$(c.contents)\n# -- end code block --")
    end
end

# Cell attributes include:
#  - hidden (boolean) - indicates if code is folded
#  - disabled (boolean) - indicates if code is disabled in Pluto
#  - mustache (boolean) - preprocess the code with the Mustache engine
#  - input=(pluto, markdown, tex) - indicates type of input cell (default pluto)
#  - output=(pluto, tex, all, none) - indicate where cell output should go

hidden(c :: Cell) =
    if c.is_code
       get(c.flags, "hidden", "false") == "true"
    else
        get(c.flags, "hidden", "true") == "true"
    end

disabled(c :: Cell) = (get(c.flags, "disabled", "false") == "true")
mustache(c :: Cell) = (get(c.flags, "mustache", "false") == "true")
input_type(c :: Cell) = get(c.flags, "input", "pluto")
output_type(c :: Cell) = get(c.flags, "output", "all")

function pluto_output(c :: Cell)
    t = output_type(c)
    t == "pluto" || t == "all"
end

function tex_output(c :: Cell)
    if hidden(c) && c.is_code
        false
    else
        t = output_type(c)
        t == "tex" || t == "all"
    end
end


# --- Parsing ---

# Parse a list of basic names (implicitly name=true) and name=value pairs.
# Folds these into an existing set of defaults.
#
function parse_mode(s, mode_defaults=Dict{String,String}())
    mode_flags = copy(mode_defaults)
    for flag in split(s, r"\s")
        if flag == ""
            continue
        end
        m=match(r"(?<key>[^=]+)=(?<val>.*)", flag)
        if m==nothing
            mode_flags[flag] = "true"
        else
            mode_flags[m[:key]] = m[:val]
        end
    end
    mode_flags
end

# Check if any flags are enabled
function parse_enabled(s, mode_flags)
    is_enabled = false
    for flag in split(s, r"\s+")
        x = get(mode_flags, flag, "false")
        is_enabled = (is_enabled || x == "true")
    end
    is_enabled
end

# Disabled if block is CIF [^CEND]* CEND
#
function process_cif(line, lines, state, mode_flags)
    is_enabled = parse_enabled(line[5:end], mode_flags)
    result = ""
    n = iterate(lines, state)
    while n != nothing
        line, state = n
        tt = toktype(line)
        if tt == CEND
            return (if is_enabled result else "" end), iterate(lines, state)
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            n = iterate(lines, state)
            result = result * line * "\n"
        end
    end
    error("Unexpected EOF while processing conditional")
end

# Regular if block is MIF [^MEND]* MEND (can nest)
#
function process_mif(line, lines, state, mode_flags)
    is_enabled = parse_enabled(line[5:end], mode_flags)
    result = ""
    n = iterate(lines, state)
    while n != nothing
        line, state = n
        tt = toktype(line)
        if tt == MEND
            return (if is_enabled result else "" end), iterate(lines, state)
        elseif tt == CIF
            if_result, n = process_cif(line, lines, state, mode_flags)
            result = result * if_result
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            result = result * line * "\n"
            n = iterate(lines, state)
        end
    end
    error("Unexpected EOF while processing conditional")
end


# Quote block is TQ [^TQ]* TQ
#    Generates a Markdown block according to current mode flags + hidden
#
function process_quote(lines, state, mode_flags)
    n = iterate(lines, state)
    result = ""
    while n != nothing
        line, state = n
        tt = toktype(line)
        if tt == TQ
            return text_cell(result, mode_flags), iterate(lines, state)
        elseif tt == CIF
            if_result, n = process_cif(line, lines, state, mode_flags)
            result = result * if_result
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            result = result * line * "\n"
            n = iterate(lines, state)
        end
    end
    error("Unexpected EOF while processing quote")
end

# Multiline comment is CSTART [^CEND]* CEND
#    Generates a TeX or Markdown block according to current mode + modifiers;
#    assumed hidden unless explicitly said otherwise
#
function process_cblock(line, lines, state, mode_flags)
    result = ""
    mode_flags = parse_mode(line[3:end], mode_flags)
    n = iterate(lines, state)
    while n != nothing
        line, state = n
        tt = toktype(line)
        if tt == CEND
            return text_cell(result, mode_flags), iterate(lines, state)
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            n = iterate(lines, state)
            result = result * line * "\n"
        end
    end
    error("Unexpected EOF while processing multi-line comment")
end

# Begin block is BEGIN [^END]* END
#    Generates a code block according to current mode
#
function process_begin(line, lines, state, mode_flags)
    result = line * "\n"
    n = iterate(lines, state)
    while n != nothing
        line, state = n
        tt = toktype(line)
        if tt == BEND
            result = result * line * "\n"
            return code_cell(result, mode_flags), iterate(lines, state)
        elseif tt == CIF
            if_result, n = process_cif(line, lines, state, mode_flags)
            result = result * if_result
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            result = result * line * "\n"
            n = iterate(lines, state)
        end
    end
    error("Unexpected EOF while processing code block")
end

# Cell block is MCELL [^MEND]* MEND
#    Generates a code block according to current mode with overrides
#
function process_cell(line, lines, state, mode_flags)
    result = ""
    mode_flags = parse_mode(line[7:end], mode_flags)
    n = iterate(lines, state)
    while (n != nothing)
        line, state = n
        tt = toktype(line)
        if tt == MEND
            return code_cell(result, mode_flags), iterate(lines, state)
        elseif tt == CIF
            if_result, n = process_cif(line, lines, state, mode_flags)
            result = result * if_result
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            n = iterate(lines, state)
            result = result * line * "\n"
        end
    end
    error("Unexpected EOF while processing #-cell")
end

# Cell block is MDCELL [^CEND]* CEND
#    Generates a code block according to current mode with overrides
#
function process_dcell(line, lines, state, mode_flags)
    result = ""
    mode_flags = copy(mode_flags)
    mode_flags["disabled"] = "true"
    mode_flags = parse_mode(line[7:end], mode_flags)
    n = iterate(lines, state)
    while (n != nothing)
        line, state = n
        tt = toktype(line)
        if tt == CEND
            return code_cell(result, mode_flags), iterate(lines, state)
        elseif tt == MIF
            if_result, n = process_mif(line, lines, state, mode_flags)
            result = result * if_result
        else
            n = iterate(lines, state)
            result = result * line * "\n"
        end
    end
    error("Unexpected EOF while processing #=cell")
end

# Process an iterator over lines, get a list of cells
#
function process_file(lines, mode_flags = Dict{String,String}())
    n = iterate(lines)
    mode_flags = copy(mode_flags)
    cells = Vector{Cell}()
    while n != nothing
        line, state = n
        t = toktype(line)
        if t == TQ
            cell, n = process_quote(lines, state, mode_flags)
            push!(cells, cell)
        elseif t == CSTART
            cell, n = process_cblock(line, lines, state, mode_flags)
            push!(cells, cell)
        elseif t == BEGIN
            cell, n = process_begin(line, lines, state, mode_flags)
            push!(cells, cell)
        elseif t == MCELL
            cell, n = process_cell(line, lines, state, mode_flags)
            push!(cells, cell)
        elseif t == MDCELL
            cell, n = process_dcell(line, lines, state, mode_flags)
            push!(cells, cell)
        elseif t == MODE
            mode_flags = parse_mode(line[7:end], mode_flags)
            n = iterate(lines, state)
        elseif t == GENERAL
            cell = code_cell(line, mode_flags)
            push!(cells, cell)
            n = iterate(lines, state)
        else
            n = iterate(lines, state)
        end
    end
    cells
end


# --- Mustache templating ---

function mustache_cell(c :: Cell)
    if c.is_code || !mustache(c)
        c
    else
        text_cell(Mustache.render(Mustache.parse(c.results), c.mode_flags),
                  mode_flags)
    end
end

mustache_cells(cells) = [mustache_cell(c) for c in cells]


# --- Plain output ---

# Print cells for diagnostics
#
function print_cells(cells)
    for cell in cells
        println(cell)
    end
end


# --- Pluto output ---

# Process Markdown to be Julia-friendly (with extra whitespace around display)
#
function markdown_to_pluto(s :: String)
    replace(s,
            "\\begin{align*}" => "\n\$\$\\begin{align*}",
            "\\end{align*}" => "\\end{align*}\$\$\n",
            "\\begin{align}" => "\n\$\$\\begin{align}",
            "\\end{align}" => "\\end{align}\$\$\n",
            r"\\begin{equation}\s*"s => "\n\$\$",
            r"\s*\\end{equation}"s => "\$\$\n",
            r"\\\[\s*" => "\n\$\$",
            r"\s*\\\]" => "\$\$\n",
            "\\bbR" => "\\mathbb{R}",
            "\\bbC" => "\\mathbb{C}",
            "\\macheps" => "\\epsilon_{\\mathrm{mach}}")
end

# Pre-process Markdown for LaTeX rendering with Pandoc
function markdown_to_pandoc(s :: String)
    replace(s,
            "\\begin{align*}" => "\$\$\\begin{align*}",
            "\\end{align*}" => "\\end{align*}\$\$",
            "\\begin{align}" => "\$\$\\begin{align}",
            "\\end{align}" => "\\end{align}\$\$",
            r"\\begin{equation}\s*"s => "\$\$",
            r"\s*\\end{equation}"s => "\$\$",
            r"\\\[\s*" => "\$\$",
            r"\s*\\\]" => "\$\$",
            "\\bbR" => "\\mathbb{R}",
            "\\bbC" => "\\mathbb{C}",
            "\\macheps" => "\\epsilon_{\\mathrm{mach}}")
end

# Wrap Markdown comment
#
function markdown_to_pluto_cell(s :: String)
    "md\"\"\"\n" * markdown_to_pluto(s) * "\"\"\""
end

# Map a cell to a Pluto notebook cell
#
function map_cell(c :: Cell)
    fold_default = "false"
    if c.is_code
        pc = Pluto.Cell(rstrip(c.contents))
    else
        pc = Pluto.Cell(markdown_to_pluto_cell(c.contents))
        fold_default = "true"
    end
    if get(c.flags, "hidden", fold_default) == "true"
        pc.code_folded = true
    end
    if get(c.flags, "disabled", "false") == "true"
        pc.metadata["disabled"] = true
    end
    pc
end

# Write a notebook
#
function write_pluto_notebook(cells, fname)
    pcells = [map_cell(c) for c in cells if pluto_output(c)]
    nb = Pluto.Notebook(pcells, fname)
    Pluto.save_notebook(nb, fname)
end


# --- LaTeX output ---

# Use Pandoc to convert Markdown to LaTeX
#
function pandoc_markdown_latex(s)

    # Preprocess input
    s = replace(s,
                r"\\begin{equation}\s*"s => "\$\$",
                r"\s*\\end{equation}"s => "\$\$",
                r"\\\[\s*" => "\$\$",
                r"\s*\\\]" => "\$\$")

    # Run it through Pandoc
    cmd = `pandoc -f markdown+raw_tex-auto_identifiers -t latex --shift-heading-level-by=-1`
    proc = open(cmd, "r+")
    writer = @async begin
        write(proc.in, s)
        close(proc.in)
    end
    reader = @async read(proc.out, String)
    wait(writer)
    result = fetch(reader)
    close(proc)

    # Postprocess
    replace(result, r"\n\n+" => "\n\n")
end

# Experimental version using the Markdown package LaTeX output code.
# Unfortunately, this seems to have a harder time with correct rendering.
#
function markdown_latex(s)
    result = Markdown.latex(Markdown.parse(markdown_to_pandoc(s)))
    replace(result,
            "\$\$\\begin{align" => "\\begin{align",
            "\\end{align}\$\$" => "\\end{align}\n",
            "\\end{align*}\$\$" => "\\end{align*}\n",
            r"\n\n+" => "\n\n")
end

# Wrap for listings
#
function code_latex(s)
    "\\begin{minted}{julia}\n" * rstrip(s) * "\n\\end{minted}\n"
end

# Write to file
#
function write_latex(cells, f :: IO)
    for c in cells
        if tex_output(c)
            it = input_type(c)
            if c.is_code
                println(f, code_latex(c.contents))
            elseif it == "pluto" || it == "markdown"
                println(f, pandoc_markdown_latex(c.contents))
            elseif it == "tex"
                println(f, c.contents)
            end
        end
    end
end

# Write to file by name
#
function write_latex(cells, fname :: String)
    open(fname, "w") do f
        write_latex(cells, f)
    end
end


# --- Command line ---

function main(args)

    pluto_name = ""
    tex_name = ""
    preamble_name = ""
    input_names = Vector{String}()
    verbose = false
    standalone = false

    # Process command line arguments
    mode_string=""
    n = iterate(args)
    while n != nothing
        arg, state = n
        if arg == "-p" && pluto_name == ""
            n = iterate(args, state)
            if n == nothing
                error("Missing argument to -p")
            end
            arg, state = n
            pluto_name = arg
        elseif arg == "-t" && tex_name == ""
            n = iterate(args, state)
            if n == nothing
                error("Missing argument to -t")
            end
            arg, state = n
            tex_name = arg
        elseif arg == "-P" && preamble_name == ""
            n = iterate(args, state)
            if n == nothing
                error("Missing preamble file name")
            end
            arg, state = n
            preamble_name = arg
        elseif arg == "-m"
            n = iterate(args, state)
            if n == nothing
                error("Missing flag assignment")
            end
            arg, state = n
            mode_string = mode_string * " " * arg
        elseif arg == "-v"
            verbose = true
        elseif arg == "-s"
            standalone = true
        elseif arg[1] != '-'
            push!(input_names, arg)
        else
            error("Error in command line -- unrecognized flag $arg")
        end
        n = iterate(args, state)
    end

    # Print command line flags
    if verbose
        println("--- Command line flags ---")
        println("mode:     $mode_flags")
        println("Pluto:    $pluto_name")
        println("LaTeX:    $tex_name")
        println("Input:    $input_names")
        println("Preamble: $preamble_name")
        println("--------------------------\n")
    end

    # Parse all input files
    mode_flags = parse_mode(mode_string)
    cells = Vector{Cell}()
    for fname in input_names
        append!(cells, process_file(eachline(fname), mode_flags))
    end

    # Apply Mustache templating
    cells = mustache_cells(cells)

    # Print a representation of the parse (useful for debugging)
    if verbose
        print_cells(cells)
    end

    # Output Pluto
    if !isempty(pluto_name)
        write_pluto_notebook(cells, pluto_name)
    end

    # Output tex
    if !isempty(tex_name)
        if standalone
            open(tex_name, "w") do f
                if preamble_name == ""
                    println(f, "\\documentclass{article}")
                else
                    open(preamble_name, "r") do pf
                        write(f, read(pf))
                    end
                end
                println(f, "\\begin{document}\n")
                write_latex(cells, f)
                println(f, "\n\\end{document}")
            end
        else
            write_latex(cells, tex_name)
        end
    end
end

main(ARGS)
