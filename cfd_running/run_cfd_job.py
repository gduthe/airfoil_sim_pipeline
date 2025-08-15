from .cfd_runner import run_cfd

def run_cfd_job(cfd_config, mesh_success=True, debug=False):
    if debug: # debug mode
        if mesh_success:
            run_cfd(cfd_config)
            return True
        else:
            return False
    else: # not debug mode, errors are excepted
        if mesh_success:
            try:
                run_cfd(cfd_config)
                return True
            except:
                return False
        else:
            return False