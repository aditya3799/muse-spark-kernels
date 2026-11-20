with open('submission.py') as f:
    text = f.read()

# Execute up to definition of _moe
loc = {}
exec("""
_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)
_MA_SFX = _MA_SFX_IF_PRMT
""" + text[text.find('def _main_defines'):text.find('def _moe_mid')], loc)

main_defs = loc['_main_defines'](loc['_MA_SFX'])
moe_defs = loc['MOE_DEFINES']
print("main_defines:", tuple(main_defs))
print("MOE_DEFINES: ", tuple(moe_defs))
print("Are they equal?:", tuple(main_defs) == tuple(moe_defs))
