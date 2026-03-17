#!/usr/bin/env python3
"""
NEXUS MQL5 Bridge Auto-Installer
Automatically compiles and installs the MQL5 bridge service
"""

import os
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_metatrader_paths():
    """Find MetaTrader 5 installation paths"""
    paths = {
        'terminal_exe': None,
        'metaeditor_exe': None,
        'services_folder': None
    }
    
    # Check common installation paths
    possible_paths = [
        r"C:\Program Files\MetaTrader 5",
        r"C:\Program Files (x86)\MetaTrader 5"
    ]
    
    for base_path in possible_paths:
        terminal_path = os.path.join(base_path, "terminal64.exe")
        metaeditor_path = os.path.join(base_path, "metaeditor64.exe")
        
        if os.path.exists(terminal_path):
            paths['terminal_exe'] = terminal_path
            logger.info(f"Found MT5 terminal: {terminal_path}")
            
        if os.path.exists(metaeditor_path):
            paths['metaeditor_exe'] = metaeditor_path
            logger.info(f"Found MetaEditor: {metaeditor_path}")
    
    # Find services folder in AppData
    appdata = os.getenv('APPDATA')
    if appdata:
        terminal_folders = []
        try:
            terminal_path = os.path.join(appdata, 'MetaQuotes', 'Terminal')
            if os.path.exists(terminal_path):
                for folder in os.listdir(terminal_path):
                    if folder.startswith('D0E8209F77C8CF37AD8BF550E51FF075'):
                        services_path = os.path.join(terminal_path, folder, 'MQL5', 'Services')
                        if os.path.exists(services_path):
                            paths['services_folder'] = services_path
                            logger.info(f"Found Services folder: {services_path}")
                            break
        except Exception as e:
            logger.error(f"Error finding services folder: {e}")
    
    return paths

def compile_mql5_service(metaeditor_exe, source_file):
    """Compile MQL5 service using MetaEditor"""
    try:
        logger.info(f"Compiling {source_file}...")
        
        # Use MetaEditor to compile
        cmd = [metaeditor_exe, '/compile:' + source_file, '/close']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Compilation successful")
            return True
        else:
            logger.error(f"❌ Compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error during compilation: {e}")
        return False

def main():
    """Main installation process"""
    logger.info("🚀 NEXUS MQL5 Bridge Auto-Installer")
    
    # Find paths
    paths = find_metatrader_paths()
    
    if not all(paths.values()):
        logger.error("❌ Could not find all required MetaTrader 5 paths")
        logger.info("Please ensure MetaTrader 5 is installed")
        return False
    
    # Source file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(current_dir, "mql5_bridge", "NEXUS_MT5_Bridge.mq5")
    target_file = os.path.join(paths['services_folder'], "NEXUS_MT5_Bridge.mq5")
    
    # Copy source file
    try:
        import shutil
        shutil.copy2(source_file, target_file)
        logger.info(f"✅ Copied source to: {target_file}")
    except Exception as e:
        logger.error(f"❌ Error copying source file: {e}")
        return False
    
    # Compile the service
    if compile_mql5_service(paths['metaeditor_exe'], target_file):
        logger.info("🎉 MQL5 Bridge service installed successfully!")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Open MetaTrader 5 desktop terminal")
        logger.info("2. Go to Tools -> Options -> Expert Advisors")
        logger.info("3. Enable 'Allow algorithmic trading' and 'Allow DLL imports'")
        logger.info("4. Go to Tools -> Services (Ctrl+M)")
        logger.info("5. Click 'Add' and select 'NEXUS_MT5_Bridge'")
        logger.info("6. Configure: WebServerURL=http://localhost:8000, Port=8000")
        logger.info("7. Start the service")
        return True
    else:
        logger.error("❌ Installation failed")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        input("\n✅ Installation complete! Press Enter to continue...")
    else:
        input("\n❌ Installation failed. Press Enter to exit...")
