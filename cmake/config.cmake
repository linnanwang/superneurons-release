##########################################################
# read config file
if (APPLE)
    set(CONFIG_FILE ${PROJECT_ROOT_DIR}/config.osx)
else ()
    set(CONFIG_FILE ${PROJECT_ROOT_DIR}/config.linux)
endif (APPLE)

file(STRINGS ${CONFIG_FILE} CONFIG)
foreach (KeyValue ${CONFIG})
    string(REGEX REPLACE "^[ ]+" "" KeyValue ${KeyValue})   # strip leading spaces
    string(REGEX REPLACE "[#].*$" "" KeyValue ${KeyValue})   # strip comments
    # skip blank line and comment line
    if (KeyValue)

        string(REGEX REPLACE "[ ]+$" "" KeyValue ${KeyValue})   # strip end spaces
        string(REGEX MATCH "^[^=]+" Key ${KeyValue})    # find variable name
        string(REPLACE "${Key}=" "" Value ${KeyValue})  # find the value

        # trim key and value
        if (Key)
            string(REGEX REPLACE "^[ ]+" "" Key ${Key})
            string(REGEX REPLACE "[ ]+$" "" Key ${Key})
        endif ()

        if (Value)
            string(REGEX REPLACE "^[ ]+" "" Value ${Value})
            string(REGEX REPLACE "[ ]+$" "" Value ${Value})
        endif ()

        set(${Key} "${Value}") # set the variable
    endif (KeyValue)
    #    message("key=${Key}, value=${Value}.")
endforeach ()

# end of read config file
##########################################################
