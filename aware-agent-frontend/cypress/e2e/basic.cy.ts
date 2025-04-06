describe('Basic End-to-End Test', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should load the application', () => {
    cy.get('body').should('exist')
  })

  it('should have the correct title', () => {
    cy.title().should('include', 'Aware Agent')
  })
}) 