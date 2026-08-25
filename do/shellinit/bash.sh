do-widget() {
  local out
  out=$(command Do --dry-run -- "$READLINE_LINE") || return
  READLINE_LINE="$out"; READLINE_POINT=${#READLINE_LINE}
}
bind -x '"\C-g": do-widget'
