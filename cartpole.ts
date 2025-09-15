import { type Entity, type Handle, InferenceComponent, MotorComponent, Util } from "prototwin";

export class CartPole extends InferenceComponent {

    // You need to create a template to set both motors
    public cartMotor: Handle<MotorComponent>;
    public poleMotor: Handle<MotorComponent>;

    // A variable to store the previous action between updates
    // It exists outside of updateAsync() so it persists between frames.
    #previousAction: number = 0.0;
    
    constructor(entity: Entity) {
        super(entity);
        // Here is where you initilize
        this.cartMotor = this.handle(MotorComponent); // Assigning the component (cart_motor)
        this.poleMotor = this.handle(MotorComponent); // Assigning the component (pole_motor)
    }

    public override async initializeAsync() {
        // Load the ONNX model from the local filesystem.
        // .onnx from main-v2 has 5 observation and the it follows the action_space range from python
        // Update this based on the whatever you named
        this.loadModelFromFile("cartpole-v2.onnx", 5, new Float32Array([-0.65]), new Float32Array([0.65]));
    }

    // updateAsync() function is stateless. 
    // Every time it runs, all of its internal variables are created fresh. 
    // If we didn't store the action outside of the function, it would be forgotten immediately.

    public override async updateAsync() {

        // Creating varibale names with the same class property names
        const cartMotor = this.cartMotor.value; // Try to get the component (cart_motor)
        const poleMotor = this.poleMotor.value; // Try to get the component (pole_motor)
        const observations = this.observations; // From `InferenceComponent` Library
        // This is the safety check that the Handle system enables!
        if (cartMotor === null || poleMotor === null || observations === null) { 
            return; 
        }

        // Populate observation array
        const cartPosition = cartMotor.currentPosition; // From `MotorComponent` Library
        const cartVelocity = cartMotor.currentVelocity; // From `MotorComponent` Library
        const poleAngularDistance = Util.signedAngularDifference(poleMotor.currentPosition, Math.PI);
        const poleAngularVelocity = poleMotor.currentVelocity;  

        // This 4 are the live measurements 
        observations[0] = cartPosition / 0.65;
        observations[1] = poleAngularDistance / Math.PI;
        observations[2] = cartVelocity;
        observations[3] = poleAngularVelocity;

        // Populate the 5th observation with the stored previous action for cartpole-v2
        // It is not a live measurement value, instead it is a recalled data
        // you remove this if you are training with main-v0 or main-v1
        observations[4] = this.#previousAction; // value = 0.0

        // Apply the actions
        const actions = await this.run(); // From `InferenceComponent` Library
        if (actions !== null) {

            // The model now controls targetPosition, not targetVelocity for cartpole-v2
            cartMotor.targetPosition = actions[0];

            // Store the new action so it can be used as the "previousAction" in the next update cycle
            // you remove this if you are training with main-v0 or main-v1
            this.#previousAction = actions[0];
        }
    }
}