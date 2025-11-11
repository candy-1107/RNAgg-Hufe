@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=D:\PYCode\RNAgg-Hufe"
set "CM_DIR=%PROJECT_ROOT%\results\alignments\cms"
set "OUTPUT_DIR=%PROJECT_ROOT%\output"
set "ALIGN_DIR=%PROJECT_ROOT%\results\alignments"

REM List of families from RF00001 to RF00010
for /l %%i in (1, 1, 10) do (
    set "num=0000%%i"
    set "FAMILY=RF!num:~-5!"

    REM List of variants and their corresponding directories
    set "VARIANTS=nuc_unaligned non-nuc_unaligned nuc_aligned non-nuc_aligned"
    for %%v in (!VARIANTS!) do (
        set "VARIANT_DIR=%%v"

        set "CM_FILE=!CM_DIR!\!FAMILY!.cm"
        set "INPUT_FASTA=!OUTPUT_DIR!\!VARIANT_DIR!\!FAMILY!.fasta"
        set "OUTPUT_STO=!ALIGN_DIR!\!VARIANT_DIR!\!FAMILY!_aligned.sto"

        if exist "!INPUT_FASTA!" (
            echo Running cmalign for !FAMILY! [!VARIANT_DIR!]
            cmalign --outformat Pfam -o "!OUTPUT_STO!" "!CM_FILE!" "!INPUT_FASTA!"
        ) else (
            echo Skipping: Input FASTA not found for !FAMILY! [!VARIANT_DIR!]
        )
    )
)

echo "--- cmalign process completed ---"
endlocal

