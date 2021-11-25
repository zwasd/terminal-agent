$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "$scriptPath\rl_strategy.py"

py -3 $algoPath
