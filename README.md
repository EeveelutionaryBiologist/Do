# Do

## What is this?

Natural Language to linux shell script parser with a rules-based safety layer, caching and inline editing. 

## Why use this? 

While other capable language-to-shell translators undeniably exist, Do explicitly supports bash, zsh and fish out of the box (auto-detected from the terminal). Furthermore, it comes with several convenience and security features, most notably: 
- Generated commands can be edited within the terminal
- auto-classification of commands by destructiveness, warnings and blast radius estimate
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
The server status can be checked via 

```
Do --status
```


## Usage 

When the daemon (dod) is running, you can run in any terminal:

```
Do "List all files in this directory"
```

The first command of each session may take a moment, as the model has to be loaded to memory.

## Background Model 

By default, the tool deploys Qwen2.5-Coder-1.5B-Instruct in a background daemon via llama-cpp - no input needed. This can be swapped out in the config, if one so wishes (Qwen2.5-Coder-0.5B-Instruct being an obvious alternative). The model will use GPU-acceleration if available and not deactivated, furthermore models will be unloaded after a set period to free resources. 


## Configuration

By default, the program looks for a config under `$XDG_CONFIG_HOME/do/config.toml` (usually `~/.config/do/config.toml`), 
but will just load standard values if none can be found. This should be fine for most use cases. If you want to change the config,
you can find an example under `Do/config/config.toml`
