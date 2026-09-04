# Do

## What is this?

Natural Language to linux shell script parser with a rules-based safety layer, caching and inline editing. 

## Why use this? 

While other capable language-to-shell translators undeniably exist, Do explicitly supports bash, zsh and fish out of the box (auto-detected from the terminal). Furthermore, it comes with several convenience and security features, most notably: 
- Generated commands can be edited within the terminal
- deterministic auto-classification of commands by destructiveness, warnings and blast radius estimate
- background caching - the model won't run twice for the same query

A specialized model fine-tuned on a diverse collection of verified bash, fish and zsh commands is currently in the works and will be served soon.

## Installation

From the repository, the easiest way is (assuming you have uv (-> https://github.com/astral-sh/uv) installed):

```
git clone https://github.com/EeveelutionaryBiologist/Do
cd Do
uv tool install ".[parse]"
Do --setup
```
And that should do it. The background service can be started via:

```
dod
```
(Should also be started automatically by the setup) 
The server status can be checked via 

```
Do --status
```


## Usage 

When the daemon (dod) is running, you can run in any terminal:

```
Do "list all python files in subdirectories"
```

The first command of each session may take a moment, as the model has to be loaded to memory.

## Background Model 

By default, the tool deploys Qwen2.5-Coder-1.5B-Instruct in a background daemon via llama-cpp - no input needed. This can be swapped out in the config, if one so wishes (Qwen2.5-Coder-0.5B-Instruct being an obvious, lighter alternative). The model will use GPU-acceleration if available and not deactivated, furthermore models will be unloaded after a set period to free resources. 


## Subprocess caveats

Generated shell commands are dispatched to and run through a detached subprocess from the Do process. This is mostly done for practical reasons but also in the name of process isolation, so that a rare malformed command does not take the main process with it. 
A side effect of this is that directory changes do not stick by default to the main shell - the change happens in the subprocess which then just returns it's exit code while the main process stays where it is. The intended way to circumvent this is by a
simple shell widget. Run:

```
Do --init-shell $SHELL | $SHELL
```

This gives you access to a shortcut (CTRL+G) to directly execute the command in-shell. 

## Configuration

By default, the program looks for a config under `$XDG_CONFIG_HOME/do/config.toml` (usually `~/.config/do/config.toml`), 
but will just load standard values if none can be found. This should be fine for most use cases. If you want to change the config,
you can find an example under `Do/config/config.toml`
