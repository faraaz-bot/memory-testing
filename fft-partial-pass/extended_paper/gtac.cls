%% This is the GTAC LaTeX class for AMD authors based on the IEEEtran cls file.
%% Contributors:
	%% Mehdi Saeedi & Gabe Loh (2024). 
\usepackage{graphicx}
\usepackage{draftwatermark}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{lipsum} % for dummy text
\usepackage{soul}
\usepackage{perpage} % Load the perpage package
\MakePerPage{footnote} % Reset the footnote counter on each page

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gtac}[Custom gtac class]

% Define a new if-switch
\newif\ifblind

% Define a new class option 'blind'
\DeclareOption{blind}{\blindtrue}

% Process the class options
\ProcessOptions\relax

% Load IEEEtran class
\LoadClass{mod_IEEEtran}

\usepackage{soul}

% Conditionally define the \author command
\ifblind
  \typeout{The 'blind' option is enabled.}
  \renewcommand{\author}[1]{}
\else
  \typeout{The 'blind' option is not enabled.}
  \renewcommand{\author}[1]{\fontsize{11}{13}\selectfont\def\@author{#1}}
\fi

\setlength{\columnwidth}{3.25in}
\setlength{\columnsep}{0.5in}

\usepackage[font={bf,small},labelfont=bf,labelsep=period]{caption} % small corresponds to 9pt in the IEEEtran class


\renewcommand{\figurename}{Figure}

\RequirePackage{titlesec}

\usepackage[left=0.75in, right=0.75in, top=1.0in, bottom=1.0in, paperheight=11in, paperwidth=8.5in, headsep=0.2in]{geometry}


% Customize title font to 18pt (Times)
\let\oldtitle\title % Store the old \title command as \oldtitle
\renewcommand{\title}[1]{\oldtitle{\fontsize{18}{21}\bfseries\selectfont #1}} % Redefine the \title command

\newcommand{\subtitle}[1]{\\[0em] \normalfont\fontsize{10}{12}\selectfont #1} % Define a new \subtitle command


\def\IEEEcompsocdiamondline{}


\renewcommand{\abstractname}{\rmfamily\fontsize{11}{13}\selectfont\bfseries\centering ABSTRACT \\ \normalfont}

\def\abstract{\normalfont
    \if@twocolumn
      \@IEEEabskeysecsize{\abstractname}\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\vspace{-1.78ex}\@IEEEabskeysecsize\textbf{\abstractname}\par\addvspace{0.5\baselineskip}\egroup\@IEEEabskeysecsize
    \fi\@IEEEgobbleleadPARNLSP}
\def\endabstract{\relax\ifCLASSOPTIONconference\vspace{0ex}\else\vspace{1.34ex}\fi\par\if@twocolumn\else\endquotation\fi
    \normalfont\normalsize}


\def\thebibliography#1{\section*{\refname}%
    \addcontentsline{toc}{section}{\refname}%
    % V1.6 add some rubber space here and provide a command trigger
    \vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep\relax
    \itemsep \IEEEbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@IEEElatexbibitem\bibitem%
    \def\bibitem{\@IEEEbibitemprefix\@IEEElatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
% originally:
%   \sloppy\clubpenalty4000\widowpenalty4000%
% by adding the \interlinepenalty here, we make it more
% difficult, but not impossible, for LaTeX to break within a reference.
% The IEEE almost never breaks a reference (but they do it more often with
% technotes). You may get an underfull vbox warning around the bibliography, 
% but the final result will be much more like what the IEEE will publish. 
% MDS 11/2000
\ifCLASSOPTIONtechnote\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty100%
\else\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty500\fi%
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist


\newcommand{\keywords}[1]{%
    \begingroup
    \let\and\\ % Handle line breaks
    \par
	\noindent\textbf{\fontsize{9}{11}\selectfont Keywords:} \fontsize{9}{11}\selectfont #1 % Body of the header
    \par
    \endgroup
}

\renewcommand{\footnotesize}{\fontsize{7pt}{8.4pt}\selectfont}
\skip\@mpfootins = \skip\footins
\fboxsep = 3pt
\fboxrule = .4pt
% V1.6 use 1em, then use LaTeX2e's \@makefnmark
% Note that the IEEE normally *left* aligns the footnote marks, so we don't need
% box resizing tricks here.
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@makefnmark}#1}% V1.6 use 1em
% V1.7 compsoc does not use superscipts for footnote marks
\ifCLASSOPTIONcompsoc
\def\@IEEEcompsocmakefnmark{\hbox{$^{\normalfont\@thefnmark}$}}
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@IEEEcompsocmakefnmark}#1}
\fi


\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

\def\section{\@startsection{section}{1}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus 1ex minus 0ex}{\fontsize{11}{13}\selectfont\rmfamily\bfseries}}%

\def\subsection{\@startsection{subsection}{1}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus 1ex minus 0ex}{\fontsize{10}{12}\selectfont\bfseries}}%

\def\subsubsection{\@startsection{subsubsection}{1}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus 1ex minus 0ex}{\fontsize{10}{12}\selectfont\textit}}%

% Now paste your additional code from the preamble here.
% Replace \usepackage with \RequirePackage. (The syntax for both commands is the same.)

% End the class definition

\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}
\renewcommand{\ttdefault}{pcr} 
\renewcommand{\familydefault}{\rmdefault}
\fontsize{10pt}{12pt}\selectfont


\endinput
