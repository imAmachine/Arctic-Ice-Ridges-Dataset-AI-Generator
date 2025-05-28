# from UI.common.base import AppArgProcessor
# from generativelib.model.enums import ExecPhase

# 
# Will in progress
# 
# class AppModeProcessor(AppArgProcessor):
#     def __init__(self, arg_value):
#         super().__init__(arg_value)
#         self.modes = {
#             ExecPhase.TRAIN.value: self._train_phase,
#             ExecPhase.TEST.value: self._test_phase,
#             ExecPhase.INFER.value: self._infer_phase
#         }

#     def _train_phase(self):
#         print(f"Обучение модели {args.model} на {args.epochs} эпохах...")
#         model.build_train_modules(model_cfg)
#         model._evaluators_from_config(model_cfg['evaluators_info'], device=DEVICE)
        
#         if args.load_weights:
#             checkpoint_path = os.path.join(WEIGHTS_PATH, 'training_checkpoint.pt')
#             load_checkpoint(model, checkpoint_path)
#         trainer.run()
    
#     def _test_phase(self):
#         pass
    
#     def _infer_phase(self):
#         pass

#     def process(self):
#         fn = self.modes(self.value)
#         fn()
