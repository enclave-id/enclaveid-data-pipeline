# Clean up unncessary files to free up space on the Actions runner
sudo rm -rf /usr/share/dotnet
sudo rm -rf /opt/ghc
sudo rm -rf "/usr/local/share/boost"
sudo rm -rf "$AGENT_TOOLSDIRECTORY"

# TODO: Decide if this is needed after a decision on infrastructure for GPU assets
# (i.e., whether to use SkyPilot or a Hybrid deployment) is made.
pip install --extra-index-url=https://pypi.nvidia.com --no-cache-dir cuml-cu12==24.2.*
