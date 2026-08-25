function do-widget
    set -l out (command Do --dry-run -- (commandline -b))
    or return
    commandline -r -- $out
    commandline -f end-of-line
end
bind \cg do-widget
