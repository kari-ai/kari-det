import threading

class Callbacks:
    def __init__(self):
        self._callbacks = {
            'on_train_start': [],
            'on_train_end': [],
            'on_train_batch_start': [],
            'on_train_batch_end': [],
            'on_train_epoch_start': [],
            'on_train_epoch_end': [],
            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],
            'on_fit_epoch_end': [],  # fit = train + val
            'on_model_save': [],
        }
        self.stop_training = False

    def register_action(self, hook, name='', callback=None):
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})


    
    def get_registered_actions(self, hook=None):
        """"
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger['callback'], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger['callback'](*args, **kwargs)

