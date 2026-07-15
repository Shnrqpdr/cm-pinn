%%%%
% rbef.cls
% Copyright (c) Youssef Cherem, 2016
% Modified 2026: removed KOMA-Script dependency for ScholarOne compatibility
% License: http://creativecommons.org/licenses/by-nc-sa/4.0/
% (CC BY-NC-SA 4.0)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rbef}[2016/04/11]

\LoadClass[a4paper,twoside]{article}

\RequirePackage{mathtools}
\RequirePackage{amssymb}
\RequirePackage{mathrsfs}

\RequirePackage[utf8]{inputenc}
\RequirePackage[english,spanish,brazil]{babel}
\RequirePackage{fourier}

\RequirePackage{textcomp}
\RequirePackage{microtype}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage[svgnames,dvipsnames]{xcolor}
\RequirePackage{sidecap}

\widowpenalty=10000
\clubpenalty=10000
\setlength{\parskip}{0pt}
\setlength{\parindent}{1.5em}

\RequirePackage[outer=2cm,inner=1.3cm,bottom=3cm,top=2.5cm]{geometry}

%% fancyhdr setup
\pagestyle{fancy}
\fancyhf{}

\RequirePackage[stable,bottom]{footmisc}
\renewcommand{\thefootnote}{\arabic{footnote}}

%%% Header/footer placeholders
\newcommand{\volumerevista}{}
\newcommand{\volume}[1]{\renewcommand{\volumerevista}{vol. #1}}

\newcommand{\numerorevista}{}
\newcommand{\numero}[1]{\renewcommand{\numerorevista}{n\textsuperscript{o} #1}}

\newcommand{\anorevista}{}
\newcommand{\ano}[1]{\renewcommand{\anorevista}{#1}}

\newcommand{\doirevista}{}
\newcommand{\doi}[1]{\renewcommand{\doirevista}{DOI: #1}}

\newcommand{\titleheader}{}
\newcommand{\titulocabecalho}[1]{\renewcommand{\titleheader}{#1}}

\newcommand{\authorheader}{}
\newcommand{\autorcabecalho}[1]{\renewcommand{\authorheader}{#1}}

\newcommand{\numerocabecalho}{}
\newcommand{\numeracao}[1]{\renewcommand{\numerocabecalho}{#1}}

\newcommand{\tipodeartigotexto}{}
\newcommand{\tipodeartigo}[1]{\renewcommand{\tipodeartigotexto}{#1}}

\newcommand{\rbef}{Revista Brasileira de Ensino de F\'{\i}sica}

%% Headers and footers
\fancyfoot[LE,RO]{\scriptsize \rbef, \volumerevista, \numerorevista, \numerocabecalho, \anorevista}
\fancyfoot[LO,RE]{\scriptsize \doirevista}
\fancyhead[C]{}
\fancyhead[RE]{\small\itshape \titleheader}
\fancyhead[LO]{\small\itshape \authorheader}
\fancyhead[LE,RO]{\small \numerocabecalho-\thepage}

\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyfoot[L]{\scriptsize Copyright by Sociedade Brasileira de F\'{\i}sica. Printed in Brazil.}
  \fancyfoot[R]{\scriptsize \rbef, \volumerevista, \numerorevista, \numerocabecalho, \anorevista}
  \renewcommand{\headrulewidth}{0pt}
}

\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

%%% Title commands
\newcommand{\titulo}[1]{\title{\LARGE #1}}
\newcommand{\subtitulo}[1]{\newcommand{\@subtitulo}{#1}}

%%% Keywords
\newcommand{\palavraschave}[1]{\newline\textbf{Palavras-chave:} #1}
\newcommand{\keywords}[1]{\newline\textbf{Keywords:} #1}
\newcommand{\palavrasclave}[1]{\newline\textbf{Palavras clave:} #1}

%%% First page environment (strip for twocolumn)
\RequirePackage{flushend}
\RequirePackage{cuted}

\newenvironment{primeirapagina}{\begin{strip}\vspace*{-2cm}\maketitle}{\end{strip}}

%%% Caption formatting
\RequirePackage{caption}
\captionsetup{font=small,labelfont={small,bf},format=plain}

\RequirePackage{wrapfig}
\RequirePackage[normalem]{ulem}
\RequirePackage{mdframed}
\surroundwithmdframed[backgroundcolor=Lavender]{verbatim}

\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{longtable}
\RequirePackage{ccicons}
%% Use url package only (hyperref removed for TeX Live 2013 compat)
\RequirePackage{url}
\providecommand{\href}[2]{#2}
\providecommand{\hypersetup}[1]{}
\RequirePackage[switch,columnwise]{lineno}
\RequirePackage[blocks]{authblk}
\renewcommand\Affilfont{\small}
\renewcommand\Authand{, }
\renewcommand\Authands{, }

\RequirePackage{abstract}

%%% Smaller bibliography font
\renewenvironment{thebibliography}[1]{%
  \section*{\refname}%
  \small
  \list{\@biblabel{\@arabic\c@enumiv}}%
       {\settowidth\labelwidth{\@biblabel{#1}}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \setlength{\parskip}{0pt}%
        \setlength{\itemsep}{0pt plus 0.3ex}%
        \@openbib@code
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}


%%% Float placement parameters
\renewcommand\floatpagefraction{.9}
\renewcommand\dblfloatpagefraction{.9}
\renewcommand\topfraction{.9}
\renewcommand\dbltopfraction{.9}
\renewcommand\bottomfraction{.9}
\renewcommand\textfraction{.1}
\setcounter{totalnumber}{50}
\setcounter{topnumber}{50}
\setcounter{bottomnumber}{50}

\setlength{\columnsep}{18pt}
\setlength{\footskip}{36pt}

\date{\vspace{-5ex}}
\renewcommand{\date}{\vspace{-5ex}}

\frenchspacing

%%% Section font: serif (like KOMA default with rmfamily)
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\normalfont\Large\bfseries}}

\RequirePackage{multirow}
\RequirePackage{enumitem}
\setlist{noitemsep}

\newmdenv[rightline=false,leftline=false,
linecolor=gray!50,linewidth=1.5pt]{equacaolonga}

%%% dblfloatfix internals (unchanged)
\newcommand*{\DBLFIXerrormessage}[1]{%
	\PackageError{dblfloatfix}{%
		Don't use the #1 package!^^J%
		You're in big trouble now...}%
	{Type `X' dammit!}%
}
\AtBeginDocument{%
	\@ifpackageloaded{stfloats}{\DBLFIXerrormessage{stfloats}}{}%
	\@ifpackageloaded{fix2col}{\DBLFIXerrormessage{fix2col}}{}%
}

\endinput
