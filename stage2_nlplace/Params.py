
import os
import sys
import json 
import math 
from collections import OrderedDict
import pdb

# #region agent log
DEBUG_LOG_PATH = None
def _debug_log(hypothesis_id, location, message, data=None):
    pass
# #endregion

class Params:
    """
    @brief Parameter class
    """
    def __init__(self, config_file=None):
        """
        @brief initialization
        @param config_file optional path to config JSON file
        """
        # #region agent log
        # _debug_log("H1", "Params.__init__:entry", "init called", {"config_file": config_file})
        # #endregion
        
        self.__dict__ = {}
        self.__dict__['params_dict'] = {}
        
        # If config_file is provided, load it directly
        if config_file is not None:
            # #region agent log
            # _debug_log("H3", "Params.__init__:load_config", "loading config file", {"config_file": config_file})
            # #endregion
            self.load(config_file)
            return
        
        # Try to load default params.json (backward compatibility)
        filename = os.path.join(os.path.dirname(__file__), 'params.json')
        # #region agent log
        # _debug_log("H1", "Params.__init__:default_file", "checking default params.json", {"filename": filename, "exists": os.path.exists(filename)})
        # #endregion
        
        if not os.path.exists(filename):
            # #region agent log
            # _debug_log("H1", "Params.__init__:no_default", "params.json not found, using empty defaults", {})
            # #endregion
            # No default file, initialize with empty defaults
            return
        
        try:
            with open(filename, "r") as f:
                params_dict = json.load(f, object_pairs_hook=OrderedDict)
            # #region agent log
            # _debug_log("H2", "Params.__init__:loaded", "loaded params.json", {"keys": list(params_dict.keys())})
            # #endregion
            
            for key, value in params_dict.items():
                if isinstance(value, dict) and 'default' in value: 
                    self.__dict__[key] = value['default']
                else:
                    # Support simple key-value format directly
                    self.__dict__[key] = value
            self.__dict__['params_dict'] = params_dict
        except json.JSONDecodeError as e:
            # #region agent log
            # _debug_log("H2", "Params.__init__:json_error", "JSON decode error", {"error": str(e)})
            # #endregion
            # Empty or invalid JSON, continue with empty defaults
            pass

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """\
========================================================
                       DREAMPlace
            Yibo Lin (http://yibolin.com)
   David Z. Pan (http://users.ece.utexas.edu/~dpan)
========================================================"""
        print(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table 
        """
        key_length = len('JSON Parameter')
        key_length_map = []
        default_length = len('Default')
        default_length_map = []
        description_length = len('Description')
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3: # python 2
                flag = isinstance(value['default'], unicode)
            else: #python 3
                flag = isinstance(value['default'], str)
            if flag and not value['default'] and 'required' in value: 
                return value['required']
            else:
                return value['default']

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value['description']))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length, description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
                'JSON Parameter', 
                " " * (key_length - len('JSON Parameter') + 1), 
                'Default', 
                " " * (default_length - len('Default') + 1), 
                'Description', 
                " " * (description_length - len('Description') + 1)
                )
        content += "| %s | %s | %s |\n" % (
                "-" * (key_length + 1), 
                "-" * (default_length + 1), 
                "-" * (description_length + 1)
                )
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                    key, 
                    " " * (key_length - key_length_map[count] + 1), 
                    str(getDefaultColumn(key, value)), 
                    " " * (default_length - default_length_map[count] + 1), 
                    value['description'], 
                    " " * (description_length - description_length_map[count] + 1)
                    )
            count += 1
        return content 

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != 'params_dict': 
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        # #region agent log
        # _debug_log("H4", "Params.fromJson:entry", "fromJson called", {"keys": list(data.keys())})
        # #endregion
        for key, value in data.items(): 
            # Support both formats:
            # 1. {"param": {"default": value, "description": "..."}}
            # 2. {"param": value}
            if isinstance(value, dict) and 'default' in value:
                self.__dict__[key] = value['default']
            else:
                self.__dict__[key] = value
        # #region agent log
        # _debug_log("H4", "Params.fromJson:exit", "fromJson completed", {"loaded_keys": [k for k in self.__dict__.keys() if k != 'params_dict']})
        # #endregion

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file
        """
        # #region agent log
        # _debug_log("H3", "Params.load:entry", "load called", {"filename": filename, "exists": os.path.exists(filename)})
        # #endregion
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))
        # #region agent log
        # _debug_log("H3", "Params.load:exit", "load completed", {"filename": filename})
        # #endregion

    def __str__(self):
        """
        @brief string
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def design_name(self):
        """
        @brief speculate the design name for dumping out intermediate solutions 
        """
        if self.aux_input: 
            design_name = os.path.basename(self.aux_input).replace(".aux", "").replace(".AUX", "")
        elif self.verilog_input:
            design_name = os.path.basename(self.verilog_input).replace(".v", "").replace(".V", "")
        elif self.def_input: 
            design_name = os.path.basename(self.def_input).replace(".def", "").replace(".DEF", "")
        return design_name 

    def solution_file_suffix(self): 
        """
        @brief speculate placement solution file suffix 
        """
        if self.def_input is not None and os.path.exists(self.def_input): # LEF/DEF 
            return "def"
        else: # Bookshelf
            return "pl"
