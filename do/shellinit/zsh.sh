do-widget() {
  local out
  out=$(command Do --dry-run -- "$BUFFER") || return
  BUFFER="$out"; CURSOR=${#BUFFER}
}
zle -N do-widget
bindkey '^G' do-widget
